from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import pandas as pd
import ast
import tqdm
import re, string
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
import pickle

def mode(x):
    z = x.value_counts()
    if len(z)==1:
        if z.item() > 1:
            return x.mode().item()
        else:
            return np.nan
    else:
        if (z.max() == z).sum()>1:
            return np.nan
        else:
            return x.mode().item()
    
def all(x):
    if x.nunique()==1:
        return x.iloc[0]
    else:
        return np.nan

def any_cl(cl):   
    def any(x):
        if np.any(x==cl):
            return cl 
        else:
            return mode(x)
    return any
     
def get_target_data(df, name, groups = ['eth_group_to_code', 'document.id'], func='mode', sense=False, save=False,
                    any_class='yes'):
    #remove or not remove texts that don't make sense
    if sense:
        df = df[df.do_text_make_sense_raw=='yes']
    #remove na #choose answer according to reviewers (or na if impossible)  
    if func == 'mode':
        df = df[pd.notnull(df[name])].groupby(groups)[name].agg(mode).dropna().to_frame().reset_index()
    elif func == 'all':
        df = df[pd.notnull(df[name])].groupby(groups)[name].agg(all).dropna().to_frame().reset_index()
    elif func == 'any':
        df = df[pd.notnull(df[name])].groupby(groups)[name].agg(any_cl(any_class)).dropna().to_frame().reset_index()
    if save:
        df.to_csv(f'data_for_"{name}".csv')
    return df[['document.id', 'eth_group_to_code', name]]

def prepare_data_single_target(config,
                 texts_path='file1_14998_texts_and_metadata.txt', 
                 coded_path ='file2_14998_coding_results.txt'):
    
    target_name = config.target_name
    #читаем coded_results и получаем данные для таргета
    coded = pd.read_csv(coded_path, delimiter='	', encoding='utf8', low_memory=False)
    data =  get_target_data(coded, target_name, func=config.func)
        
    #обрабатывать переменную как этноспецифичную (добавлять в начало текста этничность) или нет
    #читаем тексты и совмещаем с данными
    texts = pd.read_csv(texts_path, delimiter='	', encoding='utf8').drop(columns='stage').drop_duplicates().set_index('document.id')
    data = data.set_index('document.id').join(texts)
    data = data[~data[target_name].isin(config.drop)]
    
    #автоматическая кодировка (например yes и no как то случайно кодируются в 0 и 1) или своя, или без кодировки (если уже закодировано)
    if config.coding == 'auto': 
        data[target_name] = data[target_name].astype('category')
        coding = {k:v for k,v in zip(range(len(data[target_name].cat.categories)), data[target_name].cat.categories)}
        data[target_name] = data[target_name].cat.codes
    elif config.coding is not None:
        data[target_name] = data[target_name].map(config.coding).astype(int)
        coding=config.coding
    else:
        coding=None
           
    if not config.ethno_specific:
        data = data[['source_text', target_name]].dropna().drop_duplicates(subset='source_text')
    else:
        data = data[['source_text', target_name, 'eth_group_to_code']].dropna()

    data[config.target_name].value_counts().plot(kind='bar')
    plt.title('Распределение классов')
    plt.show()
        
    return data, coding

def split_train_test(data, config, split=None):
    
    if config.stratified_split is not None:
        if config.stratified_split=='class':
            stratified_split = data[config.target_name]
        elif config.stratified_split=='ethno' and config.ethno_specific==True:
            stratified_split = data['eth_group_to_code']
            single = stratified_split.value_counts()
            single = single[single==1].index
            stratified_split[stratified_split.isin(single)] = 'single_instance_class'
    else:
        stratified_split = None
    
    if not config.ethno_specific:
        train, test = train_test_split(data, test_size=split, 
                                       random_state=123, stratify=stratified_split)
    else:
        if not config.split_on_document:
            train, test = train_test_split(data, test_size=split, 
                                           random_state=123, stratify=stratified_split)
        else:
            test_index = data.index.to_series().sample(frac=split, random_state=123)
            test = data.loc[test_index]
            train = data.drop(test_index, axis=0)
        
    return train, test

def get_resampler(target, second_target=None):
    print(len(target))
    if second_target is not None:
        print(len(second_target))
        class_sample_count = second_target.value_counts().sort_index().values
        second_weight = 1. / class_sample_count 
    class_sample_count = target.value_counts().sort_index().values
    weight = 1. / class_sample_count
    if second_target is not None:
        samples_weight = np.array([weight[t]*second_weight[j] for t,j in zip(target,
                              second_target.repeat(len(target)//len(second_target)))])
        samples_weight = samples_weight/sum(samples_weight)
    else:
        samples_weight = np.array([weight[t] for t in target])
    samples_weight = torch.from_numpy(samples_weight)
    #samples_weigth = samples_weight.double()
    print(len(samples_weight))
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    return sampler
    
class EthnoHateDataset(Dataset):
    def __init__(self, data, configs=None, type='train', nli_path=None):
        self.data = data
        self.data.iloc[:,0] = self.data.iloc[:,0].apply(lambda x: str(x))
        if len(self.data.columns)>1:
            if self.data.iloc[:,1].dtype == 'O':
                self.data.iloc[:,1] = data.iloc[:,1].astype('category')
                self.data.iloc[:,1] = data.iloc[:,1].cat.codes
            self.n_classes = len(data.iloc[:, 1].unique())
        self.EthnoSpec = 'eth_group_to_code' in self.data.columns
        #self.ethnicity_processing = configs.ethnicity_processing
        self.mask = None
        self.tokenizer = None
        self.prompt = None
        self.epoch = 1
        self.weighting_type = None
        self.aug_rest = configs.aug_rest if type=='train' else False
        self.masking = configs.masking if type=='train' else 0
        self.nli = False
        if configs is not None:
            self.masking = configs.masking if type=='train' else 0
            self.idx_to_augment = torch.empty(0)
            try:
                self.nli = configs.nli
            except:
                self.nli = False
            try:
                self.reverse_weighting = configs.reverse_weighting
                self.extractor = configs.extractor
                self.replacer = configs.etno_replacer if type=='train' else False
            except:
                self.extractor = False
                self.replacer = False
            self.aug_warming_up = configs.aug_warming_up
            self.original_p = configs.original_text_p
            self.texts_info = None
            if self.masking>0:
                self.warm_up_epochs = configs.masking_warm_up
                try:
                    self.n_safe = configs.n_safe
                except:
                    self.n_safe = 0
                self.masking_type = configs.masking_type
                try:
                    self.weighting_type = configs.weighting_type
                except:
                    self.weighting_type = None
                if self.weighting_type == 'non_etno':
                    self.texts_info = pd.read_pickle(f'texts_info_simple.pickle')
                self.idx_to_mask = torch.empty(0)
            if self.nli:
                try:
                    self.nli_data = pd.read_csv(configs.nli_path, index_col=0)
                except:
                    self.labels = configs.labels
                    self.nli_template = configs.nli_template
                    nli_texts = self.data.iloc[:,0].repeat(len(self.labels))
                    nli_labels = pd.Series(self.labels*len(data))
                    nli_classes = pd.Series(pd.get_dummies(data.iloc[:,1]).to_numpy().flatten())
                    print(nli_texts.shape, nli_labels.shape, nli_classes.shape)
                    self.nli_data = pd.DataFrame({'premise':nli_texts.values, 
                                             'hypo':nli_labels.values, 
                                             'class' : nli_classes.values})
                    print(f'Shape of nli data = {self.nli_data.shape}')
            if self.replacer:
                self.replacer_regime = configs.etno_replacer_regime
                with open(f'transform_slovar_{self.replacer_regime}.pickle', 'rb') as f:
                    self.transform_slovar = pickle.load(f)
                self.texts_info = pd.read_pickle(f'texts_info_{self.replacer_regime}.pickle')
                self.replacer_warming = configs.etno_replacer_warming
                self.replace_miss_spelled = configs.replace_miss_spelled
            if self.extractor:
                if self.texts_info is None:
                    self.texts_info = pd.read_pickle(f'texts_info_simple.pickle')
                self.window = configs.extractor_window
                self.add_sep = configs.add_sep
                self.min_length = configs.min_length

    def modify_text(self):
        self.data.iloc[:,0] = self.data.iloc[:,0].apply(lambda x: self.prompt.replace('[TEXT]', x))
        if self.weighting_type in ['TF-IDF','TF']:
            for t in range(self.data.shape[0]):
                self.data.iloc[t,0] = ' '.join(list(np.array(re.findall(fr"[\w']+|[{string.punctuation}]",
                                                                         self.data.iloc[t,0]))))
            self.TFIDF = TFIDF(use_idf = True if self.weighting_type=='TF-IDF' else False,
                               token_pattern=fr"[\w']+|[{string.punctuation}]")
            self.tfidf_matrix = self.TFIDF.fit_transform(self.data.iloc[:,0])
            self.tfidf_matrix = pd.DataFrame(self.tfidf_matrix.toarray(), 
                                        columns = self.TFIDF.get_feature_names())
            self.weights = []
            for t in range(self.data.shape[0]):
                text = re.findall(fr"[\w']+|[{string.punctuation}]",
                                   self.data.iloc[t,0])
                temp = []
                for i in range(len(text)):
                    temp.append(self.tfidf_matrix.loc[t,text[i]]) 
                temp = np.array(temp)/sum(temp)
                if self.reverse_weighting:
                    temp = 1/temp
                    temp = temp/temp.sum()
                self.weights.append(temp)
        
    def __len__(self):
        if not self.nli:
            return self.data.shape[0]
        else:
            return self.nli_data.shape[0]
        
    def replace(self, text, idx):
        #print(text)
        actual_replacement = dict()
        try:
            for i in self.texts_info.loc[idx]['possible_replacements'].keys():
                actual_replacement[i] = np.random.choice(self.texts_info.loc[idx]['possible_replacements'][i])
        except:
            print(f'no text no. {idx}')
            return text
        for e,g,_,s,p in zip(*self.texts_info.loc[idx].iloc[1:6].tolist()):
            text = text.replace(e, 
                                str(np.random.choice(self.transform_slovar[actual_replacement[g]][s])[p]))
        if self.replace_miss_spelled:
            for i in self.texts_info.loc[idx]['possible_replacements_miss_spelled'].keys():
                actual_replacement[i] = np.random.choice(self.texts_info.loc[idx]['possible_replacements_miss_spelled'][i])
            for e,g,_,s,p in zip(*self.texts_info.loc[idx].iloc[1:6].tolist()):
                text = text.replace(e, 
                        str(np.random.choice(self.transform_slovar[actual_replacement[g]][s])[p]))
        #print(text)
        #print('--------------')
        return text
    
    def extract(self, text, idx):
        try:
            etnos = self.texts_info.loc[idx]['all_etnos']
        except:
            print(f'cant extract for text no. {idx}')
            return text
        text = re.findall(fr"[\w'{string.punctuation}]+", text)
        if len(text)<self.min_length:
            print(f'too short text no. {idx} !!!')
        nums = []
        for e in etnos:
            for n, t in enumerate(text):
                if e in t and n not in nums:
                    nums.append(n)
        nums = sorted(nums)
        new_text = []
        for n in range(len(nums)):
            new_text.extend(text[max(0 if n == 0 else nums[n-1],
                                nums[n]-self.window):min(nums[n]+self.window, 
                                                    len(text) if len(nums)-n==1 else nums[n+1])])
            if self.add_sep:
                new_text.extend([self.sep_token])
        text = ' '.join(new_text) 
        #print(text)
        return text


    def __getitem__(self, idx):
        if self.nli:
            text, hypo, target, *rest = self.nli_data.iloc[idx]
        elif not self.EthnoSpec:
            text, target, *rest = self.data.iloc[idx]
        else:
            text, target, ethnicity, *rest = self.data.iloc[idx]
            if self.ethnicity_processing=='internal':
                text = ethnicity + '[SEP]' + text

            elif self.ethnicity_processing == 'external': #TO-DO
                return text, ethnicity, target, rest
        
        #redundant feature
        if rest!=[] and not self.aug_rest and not self.nli:
            rest = self.proc_rest(rest)
            
        elif self.aug_rest:
            if self.epoch > self.aug_warming_up:
                p = None
                if self.original_p!='equal':
                    p = [(1-self.original_p)/len(rest) for _ in range(len(rest))] + [self.original_p]
                if self.idx_to_augment.nelement()==0:
                    text = np.random.choice(rest+[text], p=p)
                else:
                    if idx in self.idx_to_augment:    
                        text = np.random.choice(rest+[text], p=p)
        
        if self.replacer and self.epoch > self.replacer_warming:
            if not self.nli:
                text = self.replace(text, self.data.index[idx])
            else:
                text = self.replace(text, self.nli_data.index[idx])

        if self.extractor:
            if not self.nli:
                text = self.extract(text, self.data.index[idx])
            else:
                text = self.extract(text, self.nli_data.index[idx])

        if self.masking > 0 and self.epoch > self.warm_up_epochs:
            if self.idx_to_mask.nelement()==0:
                text = self.mask_text(text, 
                weights=self.weights[idx] if self.weighting_type is not None and self.weighting_type!='non_etno' else None,
                idx = self.data.index[idx])
            else:
                if idx in self.idx_to_mask:
                    text = self.mask_text(text, 
                    weights=self.weights[idx] if self.weighting_type is not None else None)
        
        if self.nli:
            #print(hypo)
            return [text, hypo], target, 'None', idx
        
        rest = 0
        return text, target, rest, idx
    
    def proc_rest(self, rest):
        if isinstance(rest[0], str):   
            rest = torch.cat([torch.tensor(ast.literal_eval(i)) for i in rest])
        else:
            rest = torch.cat([torch.tensor(i) for i in rest])
        return rest
    
    def mask_text(self, text, weights=None, idx=None):
        text = np.array(re.findall(r"[\w']+|[.,!?;:]", text), dtype='object')
        #text = np.array(re.findall(fr"[\w'{string.punctuation}]+", text), dtype='object')
        n = len(text)
        if self.weighting_type is None:
            mask = np.random.choice(range(n), size=int(n*self.masking), replace=False)
        elif self.weighting_type == 'non_etno':
            try:
                etnos = self.texts_info.loc[idx]['all_etnos']
            except:
                print(f'cant get etnos for text no. {idx}')
                text = ' '.join(list(text))
                return text
            nums = []
            safe = []
            for e in etnos:
                for n, t in enumerate(text):
                    if e in t and n not in nums:
                        safe.append(n)
                        for i in range(1,self.n_safe+1):
                            safe.append(n-i)
                            safe.append(n+i)
                    elif n not in safe:
                        nums.append(n)
            if len(nums)>0:
                mask = np.random.choice(nums, size=int(n*self.masking), replace=False, p=weights)
            else:
                text = ' '.join(list(text))
                return text
        elif self.weighting_type in ['TF-IDF','TF']:
            mask = np.random.choice(range(n), size=int(n*self.masking), replace=False, p=weights)
        if self.masking_type == 'whole words':
            for i in mask:
                if 'mtk' not in text[i] and (text[i] not in self.prompt if self.prompt is not None else True):
                    text[i] = self.mask*(len(self.tokenizer([text[i]]).input_ids[0])-2)
        else:
            cond = ['mtk' not in text[i] for i in mask]
            text[mask[cond]] = self.mask
        text = ' '.join(list(text))

        return text
  
def awmpler(target):

    class_sample_count = target.value_counts().sort_index().values
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    samples_weight = torch.from_numpy(samples_weight)
    #samples_weigth = samples_weight.double()
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
    
    return sampler

#перед этим нужно изменить кодировку на какие то большие желательно нечетные цифры и заново получить данные
def check_leak(train_data, test_data, coding):
    target = train_data.columns[1]
    inters = list(set(train_data.index).intersection(set(test_data.index)))
    same_train = train_data[target][inters].to_frame().reset_index().groupby('document.id')[target].mean()
    same_train = same_train[same_train.isin(coding.values())].index
    same_test = test_data[target][inters].to_frame().reset_index().groupby('document.id')[target].mean()
    same_test = same_test[same_test.isin(coding.values())].index
    same = list(set(same_train).intersection(set(same_test)))
    a = train_data[target].loc[same].to_frame().reset_index().drop_duplicates()
    a = a.loc[a['document.id'].drop_duplicates().index]
    b = test_data[target].loc[same].to_frame().reset_index().drop_duplicates()
    b = b.loc[b['document.id'].drop_duplicates().index]
    index = a.reset_index()[target] == b.reset_index()[target]
    print(f'{sum(index)} из {len(inters)} документов, которые встерчаются в обоих выборках и имеют одинаковый лейбл в трейне имеют такой же лейбл и в тесте')
    print(f'{sum(index)} из {len(test_data)} строчек в тесте совпадает с строчками в трейне (без учета эничности)')
    pl = a.reset_index()[target][index].value_counts().reset_index()[target]
    print(pl)
    pl.plot(kind='bar')
    plt.title('распределение классов для этих строчек')

def clean(text):
    text = text.apply(lambda x: str(x))
    text = text.apply(lambda x: x.lower())
    print('Cleaning dataset...')
    print('Removing HTML...')
    CLEANR = [re.compile('<.*?>'), re.compile("\[.*?\]")]
    for i in CLEANR:
        text = text.apply(lambda x: re.sub(i, '', x))
    text = text.apply(lambda x: x.replace('\\', ''))
    print('Replacing repetitive punctuation...')
    text = text.apply(lambda x: re.sub(r"([" + re.escape(string.punctuation) + r"])\1+", r"\1", x))
    print('Removing https...')
    text = text.apply(lambda x: re.sub(r"http\S+", '', x))
    text = text.apply(lambda x: re.sub(r"\r", '', x))
    text = text.apply(lambda x: re.sub(r'\s+', ' ', x))
    print('Removing numbers...')
    text = text.apply(lambda x: re.sub(r' *[0-9] *', ' ', x))
    print('Replacing repetitive punctuation once more...')
    text = text.apply(lambda x: re.sub(r"([" + re.escape(string.punctuation) + r"])\1+", r"\1", x))
    print('Replacing redundant  spaces...')
    text = text.apply(lambda x: re.sub(r'\s+', ' ', x))
    text = text.apply(lambda x: re.sub('"+','"', x))
    text = text.apply(lambda x: re.sub("'+","'", x))
    #text = text.apply(lambda x: x.lstrip())
    return text