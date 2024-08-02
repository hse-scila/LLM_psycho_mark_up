import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoModelForCausalLM
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, AutoModelForMaskedLM, T5ForConditionalGeneration 
from transformers import T5ForConditionalGeneration, AutoModelForSeq2SeqLM
import tqdm
import pandas as pd
from torch.nn import functional as F
import numpy as np
import re
import os

class Augmentizer():
    #для суперкомпьютера лучше загрузить с гитхаба и указать путь
    def __init__(self, data, configs, tpe='train', local=True):  
        
        self.device = configs.aug_device
        self.configs = configs
        self.type = tpe
        self.local = local
        
        self.paraphraser = configs.paraphraser
        self.grams = configs.grams
        self.encoder_grams = configs.encoder_grams
        self.paraphraser_task = configs.paraphraser_task
        self.gpt_restorer = configs.gpt_restorer
        self.gpt_fraction = configs.gpt_fraction
        self.back_translator = configs.translator
        self.translator_lang = configs.translator_lang
        self.imputer = configs.imputer
        self.imputer_masking = configs.imputer_masking
        self.imputer_masking_type = configs.imputer_masking_type
        self.FRED = configs.FRED
        self.top_k = configs.top_k
        self.FRED_regime = configs.FRED_regime
        self.FRED_impute = configs.FRED_impute
        self.FRED_n_mask = configs.FRED_n_mask
        self.FRED_fraction = configs.FRED_fraction
        self.FRED_n_to_add = configs.FRED_n_to_add
        self.FRED_add_random = configs.FRED_add_random
        self.FRED_threshold = configs.FRED_threshold
        self.FRED_sep = configs.FRED_sep
        self.FRED_weighted_sampling = False
        self.repetition_penalty = configs.repetition_penalty
        self.diversity_penalty = configs.diversity_penalty
        self.temperature=configs.temperature
        self.n_samples = configs.n_samples
        self.length_multiplier = configs.length_multiplier
        self.max_length = configs.max_length 
        self.beams = configs.num_beams
        self.num_beam_groups = configs.num_beam_groups
        self.do_sample = configs.do_sample
        self.alpha_penalty = configs.alpha_penalty
        
        self.aug_list = [self.imputer, self.gpt_restorer, self.paraphraser, 
                        self.back_translator, self.FRED]
        self.aug_funcs = [self.mask_unmask, self.gpt_restore_unbatched, self.paraphrase,
                        self.back_translate, self.FRED_restore]
        temp = ''
        if self.grams is not None:
            temp = f'gr_{self.grams}_'
        if self.encoder_grams is not None:
            temp = f'{temp}en_gr_{self.encoder_grams}_'
        if self.repetition_penalty is not None:
            temp = f'{temp}rep_{self.repetition_penalty}_'
        if self.diversity_penalty is not None:
            temp = f'{temp}div_{self.diversity_penalty}_'
        if self.temperature is not None:
            temp = f'{temp}t_{self.temperature}_'
        if self.top_k is not None:
            temp = f'{temp}top_{self.top_k}_'
        if self.beams is not None:
            temp = f'{temp}b_{self.beams}_'
        if self.num_beam_groups is not None:
            temp = f'{temp}bg_{self.num_beam_groups}_'
        if self.do_sample:
            temp = f'{temp}sample_'
        if self.alpha_penalty is not None:
            temp = f'{temp}contrastive_'
        if self.length_multiplier is not None:
            temp = f'{temp}l_{self.length_multiplier}_'
        elif self.max_length is not None:
            temp = f'{temp}l_{self.max_length}_'

        fred=''
        if self.FRED is not None:
            fred = f'{self.FRED_regime}_'
            if self.FRED_impute:
                fred = f'{fred}impute_frac_{self.FRED_fraction}_with_{self.FRED_n_mask}_by_{self.FRED_n_to_add}_'
                if self.FRED_add_random:
                    fred = f'{fred}random_'
                if self.FRED_sep!='':
                    fred = f'{fred}random_sep_'
            else:
                fred = f'{fred}_{self.FRED_threshold}' 
                
        self.settings = [f'mask_{self.imputer_masking}_type_{self.imputer_masking_type}', 
                         f'frac_{self.gpt_fraction}_{temp}', 
                         f'{temp}{self.paraphraser_task if self.paraphraser_task!="" else "NONE"}', 
                         self.translator_lang, f'{fred}{temp}']
        self._m = self.aug_list
        
        self.additional_features=0
        self.toxicity = configs.toxicity
        self.sentiment = configs.sentiment    
        self.emotion = configs.emotion
        self.scores_list = [self.toxicity, self.sentiment, self.emotion]
        self.data = data.sort_index()
        if 'eth_group_to_code' in self.data.columns:
            self.ethnic = True
        else:
            self.ethnic = False

        self.augment_classes = configs.aug_classes
        self.aug_rest = configs.aug_rest
        self.only_augs = configs.only_augs

        self.preprocessing_translator = configs.preprocessing_translator
        self.preprocessing_lang = configs.preprocessing_lang
        self.preprocessing_translator_batch = configs.preprocessing_translator_batch
        self.keep_in_foreign = configs.train_in_foreign
        self.postprocessing_translator = configs.postprocessing_translator

        self.add = 'orig'
        if self.preprocessing_translator is not None:
            self.add = f'{self.preprocessing_translator.split("/")[-1]}_{self.preprocessing_lang}'
            if not self.keep_in_foreign:
                if self.postprocessing_translator is not None:
                    self.add = f'{self.add}_ru_{self.postprocessing_translator.split("/")[-1]}'
                else:
                    self.add = f'{self.add}_ru'

        self.controller = configs.aug_controller
        self.controller_regime = configs.aug_controller_regime
        self.controller_max = configs.aug_controller_max
        self.controller_max_diff = configs.controller_max_diff
        self.true_etno_share = configs.true_etno_share
        self.n = self.data.shape[0]
        self.check_etnonyms = configs.check_etnonyms

        self.path = f'{self.configs.workspace}/{self.configs.target_name}/cache/{self.type}'
        self.batch_size = configs.augmentation_batch
    
    def paraphrase(self, text, iter): 
        if iter == 0:
            self.current_tokenizer = AutoTokenizer.from_pretrained(self.paraphraser, 
                                                                       local_files_only=self.local)
            self.current_model = AutoModelForSeq2SeqLM.from_pretrained(self.paraphraser, 
                                                            local_files_only=self.local).to(self.device)
            self.current_model.eval()
        if self.paraphraser_task!='':
            text = [f'{self.paraphraser_task}: {i}' for i in text]
        encoded_input = self.current_tokenizer(text, return_tensors='pt', 
                                               padding=True, max_length=512,
                                               truncation=True).to(self.device)
        max_length = int(encoded_input.input_ids.shape[1] * self.length_multiplier + 5) if self.length_multiplier is not None else self.max_length
        out = self.generate(encoded_input, max_length)
        return self.current_tokenizer.batch_decode(out, skip_special_tokens=True)
    
    def gpt_restore_unbatched(self, text, iter):
        if iter == 0:
            self.current_tokenizer = AutoTokenizer.from_pretrained(self.gpt_restorer, 
                                                               local_files_only=self.local)
            self.current_model = AutoModelForCausalLM.from_pretrained(self.gpt_restorer, 
                                                                  local_files_only=self.local).to(self.device)
            self.current_model.eval()
            self.current_tokenizer.pad_token = self.current_tokenizer.eos_token
            self.current_model.config.pad_token_id = self.current_model.config.eos_token_id

        text = text.split()
        n = len(text)
        if n > 32:
            cut = int(self.gpt_fraction*len(text))
            corrupted_text = ' '.join(text[:cut])
            text_to_restore = ' '.join(text[cut:])
        else:
            corrupted_text = ' '.join(text)
            text_to_restore = corrupted_text
            encoded_input = self.current_tokenizer(corrupted_text, return_tensors='pt', 
                                           add_special_tokens=False, max_length=512).to(self.device)
            max_length = int(len(self.current_tokenizer(text_to_restore, return_tensors='pt', 
                            add_special_tokens=False)["input_ids"])*self.length_multiplier+5)
            out = self.generate(encoded_input, max_length)
        return self.current_tokenizer.batch_decode(out, skip_special_tokens=True)
        
    def FRED_restore(self, text, iter, weights=None):
        if iter==0:
            self.current_tokenizer = AutoTokenizer.from_pretrained(self.FRED)
            self.current_model = T5ForConditionalGeneration.from_pretrained(self.FRED).to(self.device) 
            self.current_model.eval() 
        if self.FRED_impute:
            for i in range(len(text)):
                temp = np.array(text[i].split(), dtype="object")
                n = len(temp)
                try:
                    if not self.FRED_weighted_sampling:
                        indexes = np.random.choice(range(np.random.randint(self.FRED_n_mask),
                                               n-self.FRED_n_mask+1,self.FRED_n_mask), 
                                               int(self.FRED_fraction*n), replace=False)
                    else:
                        indexes = np.random.choice(range(np.random.randint(self.FRED_n_mask),
                                               n-self.FRED_n_mask+1,self.FRED_n_mask), 
                                               int(self.FRED_fraction*n), replace=False,
                                               p=weights)
                    indexes.sort()
                    if not self.FRED_add_random:
                        #print([self.FRED_sep.join([f'<extra_id_{k*self.FRED_n_to_add+d}>' for d in range(self.FRED_n_to_add)]) for k in range(len(indexes))])
                        temp[indexes] =  [self.FRED_sep.join([f'<extra_id_{k*self.FRED_n_to_add+d}>' for d in range(self.FRED_n_to_add)]) for k in range(len(indexes))]
                    else:
                        temp[indexes] =  [self.FRED_sep.join([f'<extra_id_{k*self.FRED_n_to_add+d}>' for d in range(np.random.randint(self.FRED_n_to_add))]) for k in range(len(indexes))]
                    if self.FRED_n_mask>1:
                        for j in range(1,self.FRED_n_mask):
                            temp[indexes+j] = 'to_remove'
                        temp = temp[temp!='to_remove']
                    text[i] = ' '.join(list(temp))
                except:
                    #print(text[i])
                    print("Can't change current text!")
        else:
            for i in range(len(text)):
                temp = text[i].split()
                if len(temp)>self.FRED_threshold:
                    cut = int(self.FRED_fraction*len(temp))
                    temp = ' '.join(temp[:cut])
                    text[i] = temp
                        
        new_text = [f'{self.FRED_regime}{i}' for i in text]
        encoded_inputs = self.current_tokenizer.batch_encode_plus(new_text,max_length=512, 
                                                        padding="longest", truncation=True, 
                                                        return_tensors="pt").to(self.device)
        out = self.generate(encoded_inputs, self.max_length)
        out = self.current_tokenizer.batch_decode(out, skip_special_tokens=True)

        final_text = []
        if not self.FRED_impute:
            for i in range(len(text)*self.n_samples):
                final_text.append(text[i//self.n_samples] + out[i])
        else:
            for i in range(len(text)*self.n_samples):
                if '<extra_id' in text[i//self.n_samples]:
                    out[i] = out[i].replace("\n"," ")
                    out[i] = [re.sub('<extra_id_.+?>','',i) for i in re.findall('<extra_id_.+?>[^<]*',
                                                                             out[i])]
                    temp = text[i//self.n_samples]
                    for n, new in enumerate(out[i]):
                        temp = temp.replace(f'<extra_id_{n}>', new)
                    final_text.append(temp)
                else:
                    final_text.append(text[i//self.n_samples])
        return final_text
    def back_translate(self, text, iter):
        if iter==0:
            if 'm2m' in self.back_translator:
                self.current_model = M2M100ForConditionalGeneration.from_pretrained(self.back_translator, 
                                                            local_files_only=self.local).to(self.device)
                print('using m2m translator')
            elif 'bart' in self.back_translator:
                print('using bart translator')
                self.current_model = AutoModelForSeq2SeqLM.from_pretrained(self.back_translator, 
                                                            local_files_only=self.local).to(self.device)
            self.current_tokenizer = AutoTokenizer.from_pretrained(self.back_translator, 
                                                                             local_files_only=self.local)
            self.current_tokenizer.src_lang = 'ru'
            self.current_model.eval()
        try:
            lang_id = self.current_tokenizer.get_lang_id(self.translator_lang)
            ru_id = self.current_tokenizer.get_lang_id('ru')
        except:
            lang_id = self.current_tokenizer.lang_code_to_id[self.translator_lang]
            ru_id = self.current_tokenizer.lang_code_to_id["ru_RU"]

        generated_tokens = self.current_tokenizer(text, return_tensors="pt", 
                                                  max_length=512, padding=True).to(self.device)
        generated_tokens = self.current_model.generate(**generated_tokens, 
            forced_bos_token_id=lang_id, 
            no_repeat_ngram_size = self.grams,
            max_new_tokens=512)
        generated_tokens = self.current_model.generate(generated_tokens, 
            forced_bos_token_id=ru_id, max_new_tokens=512,
            no_repeat_ngram_size = self.grams)
        return self.current_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
  
    def mask_unmask(self, text, iter):
        if iter == 0:
            self.current_tokenizer =  AutoTokenizer.from_pretrained(self.imputer,
                                                                    local_files_only=self.local)
            self.current_model = AutoModelForMaskedLM.from_pretrained(self.imputer,
                                                            local_files_only=self.local).to(self.device)
            self.current_model.eval()
        
        text = np.array(re.findall(r"[\w']+|[.,!?;:]", text))
        n = len(text)
        mask = np.random.choice(n, size=int(n*self.imputer_masking), replace=False)
        if self.imputer_masking_type == 'whole words':
            for i in mask:
                n_tokens = len(self.current_tokenizer([text[i]]).input_ids[0])-2
                text[i] = self.current_tokenizer.mask_token*(n_tokens)
        else:
            text[mask] = self.current_tokenizer.mask_token
        text = ' '.join(list(text))
       
        inputs = self.current_tokenizer(text, return_tensors="pt").to(self.device)
        mask_token_index = torch.where(inputs["input_ids"][0] == self.current_tokenizer.mask_token_id)[0].to(self.device)
        logits = self.current_model(**inputs).logits
        mask_token_logits = logits[0, mask_token_index, :]
        pred_tokens = torch.argmax(mask_token_logits, dim=1)
        for i in mask_token_index:
            inputs["input_ids"][0][i] = pred_tokens[i]
        text = self.current_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        
        return text
        
    def text2class_scores(self, text, aggregate=False):
        
        inputs = self.current_tokenizer(text, return_tensors='pt', 
                                        truncation=True, padding=True).to(self.device)
        proba = torch.sigmoid(self.current_model(**inputs).logits).cpu()
        if isinstance(text, str):
            proba = proba[0]
        if aggregate:
            return 1 - proba.T[0] * (1 - proba.T[-1])
        
        return proba
    #!REDUNDANT FEATURE 
    def get_all_scores(self, name):
        self.data[f'{name}_score'] = None
        scores = []
        for i in tqdm.notebook.tqdm(self.data.index, desc=f'adding {name}'):
            s = self.data.loc[i]
            if isinstance(s.source_text, str):
                text = s.source_text
                score = self.text2class_scores(text)
                scores.append(score.tolist())
            else:
                text = s.source_text.iloc[0]
                score = self.text2class_scores(text)
                scores.extend([score.tolist()]*len(s))
        self.data[f'{name}_score'] = scores 
    def generate(self, inp, max_length=None):
        with torch.inference_mode():
            out = self.current_model.generate(**inp,
                                encoder_no_repeat_ngram_size = self.encoder_grams,
                                no_repeat_ngram_size=self.grams,
                                max_new_tokens = max_length,
                                num_beams=self.beams, 
                                num_return_sequences=self.n_samples,
                                num_beam_groups = self.num_beam_groups, 
                                temperature=self.temperature,
                                do_sample=self.do_sample, 
                                diversity_penalty = self.diversity_penalty,
                                repetition_penalty = self.repetition_penalty,)
        return out
 
    def add_all_rows(self, aug, func, indexes):
        for iter,i in tqdm.notebook.tqdm(enumerate(range(0,len(indexes), self.batch_size)),
                                         desc=f'adding texts for {aug.split("/")[-1]}'):
            index = indexes[i:i+self.batch_size]
            texts = self.data.loc[index].iloc[:,0].values.tolist()
            labels = self.data.loc[index].iloc[:,1].repeat(self.n_samples).values
            texts = [str(i) for i in texts]
            new_texts = func(texts, iter)
            new_rows = pd.DataFrame({self.data.columns[0]: new_texts, 
                                 self.configs.target_name : labels},
                                 index = index.repeat(self.n_samples))
            self.data = pd.concat([self.data.loc[:], new_rows.iloc[:]]) 

    def check_augs(self, indexes):
        
        self.current_model = torch.load(self.controller)
        self.current_model.eval()
        
        old_scores = []
        new_scores = []
        true_labels = torch.tensor(self.data.iloc[:self.n,1].loc[indexes].repeat(self.n_samples).values)

        with torch.inference_mode():
            for i in tqdm.notebook.tqdm(range(0, len(indexes), self.batch_size), desc='checking'):
                inp = self.data.iloc[:self.n,0].loc[indexes[i:i+self.batch_size]].tolist()
                inp = [str(i) for i in inp]
                x,_,_ = self.current_model(inp,None)
                x = F.softmax(x, dim=-1).detach().cpu()
                old_scores.append(x)
                if not self.aug_rest:
                    inp = self.data.iloc[self.n:,0].loc[indexes[i:i+self.batch_size]].tolist()
                    inp = [str(i) for i in inp]
                    x,_,_ = self.current_model(inp, None)
                else:
                    inp = self.data.iloc[:,
                                         2:].loc[indexes[i:i+self.batch_size]].values.flatten().tolist()
                    inp = [str(i) for i in inp]
                    x,_,_ = self.current_model(inp, None)
                x = F.softmax(x, dim=-1).detach().cpu()
                new_scores.append(x)
        old_scores = torch.cat(old_scores)
        print(self.n_samples)
        old_scores = torch.repeat_interleave(old_scores, self.n_samples, dim=0)
        print(old_scores.shape)
        new_scores = torch.cat(new_scores)

        cond = new_scores.argmax(dim=1)==new_scores.argmax(dim=1)
        if 'True' in self.controller_regime:
            cond = torch.logical_and(cond, new_scores.argmax(dim=1) == true_labels)
        if 'False' in self.controller_regime:
            cond = torch.logical_and(cond, new_scores.argmax(dim=1) != true_labels)
        if 'Same' in self.controller_regime :
            cond = torch.logical_and(cond, old_scores.argmax(dim=1) == new_scores.argmax(dim=1))
        if 'Opposite' in self.controller_regime:
            cond = torch.logical_and(cond, old_scores.argmax(dim=1) != new_scores.argmax(dim=1))
        if 'Max' in self.controller_regime:
            cond = torch.logical_and(cond, new_scores.max(dim=1)[0] < self.controller_max)
        if 'Min' in self.controller_regime:
            cond = torch.logical_and(cond, new_scores.max(dim=1)[0] > self.controller_max)
        if 'Diff' in self.controller_regime:
            cond = torch.logical_and(cond,
            torch.abs(new_scores[:,0] - old_scores[:,0]) < self.controller_max_diff)
        cond = ~cond
        print(f'Deleting {cond.float().mean()} augmented observations')
        cond = cond.tolist()
    
        if not self.aug_rest:
            self.data.iloc[self.n:,:].drop(indexes[cond], inplace=True)
        else:
            for i in range(self.data.shape[1]-2):
                print(self.data.iloc[:,
                2+i].loc[indexes].loc[cond[i::self.n_samples]])
                self.data.iloc[:,
                2+i].loc[indexes[cond[i::self.n_samples]]] = self.data.iloc[:,
                                            0].loc[indexes[cond[i::self.n_samples]]]
                print(self.data.iloc[:,
                2+i].loc[indexes].loc[cond[i::self.n_samples]])
                
                

    def check_etnos(self, indexes):
        self.slovar = pd.read_csv('full_slovar_last.csv', index_col=0)
        self.texts_info =  pd.read_pickle(f'texts_info_simple.pickle')
        cond = []
        for n,i in tqdm.notebook.tqdm(enumerate(pd.Series(indexes).repeat(self.n_samples).values),
                                      desc = 'checking etnhnicities'):
            try:
                if not self.aug_rest:
                    texts = [self.data.iloc[self.n:,0].loc[i]]
                    print(texts)
                else:
                    texts= self.data.iloc[:,2:].loc[i].values.flatten().tolist()
                for t in texts:
                    groups = self.find_etnos_groups(t)
                    inter = set(groups).intersection(set(self.texts_info.etnonym_groups.loc[i]))
                    if len(inter)>=len(set(self.texts_info.etnonym_groups.loc[i]))*self.true_etno_share:
                        cond.append(False)
                    else:
                        cond.append(True)
            except:
                print('cant findd text')
                cond.append(True)
        print(len(cond))
        print(f'Deleting {sum(cond)/len(cond)} augmented observations')
        if not self.aug_rest:
            print(self.data.shape)
            self.data.iloc[self.n:,:].drop(indexes[cond], inplace=True)
            print(self.data.shape)
        else:
            for i in range(self.data.shape[1]-2):
                self.data.iloc[:,
                2+i].loc[indexes[cond[i::self.n_samples]]] = self.data.iloc[:,
                                            0].loc[indexes[cond[i::self.n_samples]]]
                

    def find_etnos_groups(self, text):
        
        text = re.findall(r"\b[^ ]+?\b",text)
        groups = []
        for m,etno in enumerate(self.slovar.iloc[:, 6:].values.flatten()):
            if etno in ['там', 'тат'] or not isinstance(etno, str):
                continue
            if etno in text and etno not in groups:
                f = m//12
                groups.append(self.slovar.iloc[f,5])
        return groups
                                

    def preprocessing_translate(self, column, src, lang, desc):
        if desc=='Post' and self.postprocessing_translator is not None:
            translator = self.postprocessing_translator
        else:
            translator = self.preprocessing_translator
        self.current_tokenizer = AutoTokenizer.from_pretrained(translator, local_files_only=self.local)
        self.current_model = AutoModelForSeq2SeqLM.from_pretrained(translator,
                                                            local_files_only=self.local).to(self.device)
        self.current_model.eval()
        self.current_tokenizer.src_lang = src
        res = []
        for i in tqdm.notebook.tqdm(range(0,len(column), self.preprocessing_translator_batch),
                                        desc = f'{desc}processing translation...'):
            text = column.iloc[i:i+self.preprocessing_translator_batch].tolist()
            with torch.inference_mode():
                text = [str(i) for i in text]    
                generated_tokens = self.current_tokenizer(text, return_tensors="pt", max_length=512,
                                                          padding=True, truncation=True).to(self.device)
                
                try:
                    generated_tokens = self.current_model.generate(**generated_tokens, 
                    forced_bos_token_id=self.current_tokenizer.get_lang_id(lang), 
                    max_length=512)
                except:
                    generated_tokens = self.current_model.generate(**generated_tokens,  
                    max_length=512)
                text = self.current_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True,
                                                   max_length=512)
                res.extend(text)
        return res
    #gathers all augs
    def augment(self):
        add_texts = False
        #добавляет новые строки только к тренировочному датасету
        if self.preprocessing_translate is not None and (self.type=='train' or self.keep_in_foreign):
            if self.preprocessing_translator is not None:  
                #if True:
                try:
                    self.translated_data = pd.read_csv(f'{self.path}/data_no_augs_{self.add.split("_ru")[0]}.csv', 
                                                       index_col=0)
                    print(f'Using cached translated {self.type} data for {self.preprocessing_translator}')
                #else:
                except:
                    print(f'Cant find translated {self.type} data. Translating with {self.preprocessing_translator}...')
                    self.translated_data = self.data.copy()
                    self.translated_data.iloc[:,0] = self.preprocessing_translate(self.data.iloc[:,0], 
                                                                        'ru', self.preprocessing_lang, 
                                                                        'Pre')   
                    self.translated_data.to_csv(f'{self.path}/data_no_augs_{self.add.split("_ru")[0]}.csv')
                self.data, self.translated_data = self.translated_data.copy(), self.data.copy()
        if self.type == 'train':
            if self.augment_classes is None:
                indexes = self.data.index.unique()
            else:
                indexes = self.data.index[self.data.iloc[:,1].isin(self.augment_classes)].unique()
            for k, func, set in zip(self.aug_list, self.aug_funcs, self.settings):
                if k is not None:
                    add_texts = True
                    os.makedirs(f'{self.path}/augs/{k.split(r"/")[-1]}', exist_ok=True)
                    path = f'{self.path}/augs/{k.split(r"/")[-1]}/data_{set}_{self.add}_{self.n_samples}.csv'
                    try:     
                        temp = pd.read_csv(path, index_col=0)
                        if not self.aug_rest:
                            self.data = pd.concat([self.data.iloc[:],temp.loc[indexes].iloc[:]])
                        else:
                            for i in range(self.n_samples):
                                if self.preprocessing_translator is None or self.keep_in_foreign:
                                    self.data[f'{k.split("/")[-1]}_{i}'] = self.data.iloc[:,0]
                                else:
                                    self.data[f'{k.split("/")[-1]}_{i}'] = self.translated_data.iloc[:,0]
                                self.data[f'{k.split("/")[-1]}_{i}'].loc[indexes] = temp.iloc[:,0].loc[indexes][i::self.n_samples]
                        print(f'using cached augs for {k} {self.add} augmentation')
                    except:
                        self.add_all_rows(k, func, indexes)
                        del self.current_model
                        if self.preprocessing_translator is not None and not self.keep_in_foreign:
                            self.data.iloc[-len(indexes)*self.n_samples:,:].to_csv(path.split('_ru')[0])
                            self.data.iloc[-len(indexes)*self.n_samples:,0] = self.preprocessing_translate(
                                                                        self.data.iloc[-len(indexes)*self.n_samples:,0], 
                                                                        self.preprocessing_lang,
                                                                        'ru', 'Post')    
                            del self.current_model
                        torch.cuda.empty_cache()    
                        self.data.iloc[-len(indexes)*self.n_samples:,:].to_csv(path)
                        if self.aug_rest:
                            self.data = self.data.iloc[:len(indexes),:]
                            temp = pd.read_csv(path, index_col=0)
                            for i in range(self.n_samples):
                                self.data[f'{k.split("/")[-1]}_{i}'] = self.data.iloc[:,0]
                                self.data[f'{k.split("/")[-1]}_{i}'].loc[indexes] = temp.iloc[:,0].loc[indexes][i::self.n_samples]
            if self.preprocessing_translator is not None and not self.keep_in_foreign:
                self.data.iloc[:self.n,0] = self.translated_data.iloc[:self.n,0]
            if self.controller is not None:
                self.check_augs(indexes)
            if self.check_etnonyms:
                self.check_etnos(indexes)
        #REDUNDANT feature          
        add = ''
        if self.type=='train' and add_texts:
            add = '_'.join([k.split(r"/")[-1] if k is not None else 'None' for k in self.aug_list]) +'_' 
        
        for k, name in zip(self.scores_list, ['toxicity', 'sentiment', 'emotion']):
            if k is not None:
                path = f'{self.path}/{add}{name}.csv'
                try:
                    self.data[f'{name}_score']=pd.read_csv(path, index_col=0)
                except:
                    self.current_tokenizer = AutoTokenizer.from_pretrained(k, 
                                               local_files_only = self.local)
                    self.current_model = AutoModelForSequenceClassification.from_pretrained(k, 
                                                local_files_only = self.local).to(self.device)
                    self.current_model.eval()
                    self.get_all_scores(name)
                    self.data[f'{name}_score'].to_csv(path)
                    del self.current_model

        if self.only_augs:
            assert False
        return self.data, self.additional_features
