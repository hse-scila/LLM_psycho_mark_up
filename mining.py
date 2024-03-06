import torch
import os
import tqdm
import string
from torch.utils.data import DataLoader
from data_utils import EthnoHateDataset
import pandas as pd
import pickle
import numpy as np
import re
from torch.nn import functional as F
from data_utils import clean

class EthnoHateMinerDataset(EthnoHateDataset):
    
    def __init__(self, data, type='mine', cleaning=False):
        super().__init__(data, type=type)
        if cleaning:
            #print(type(self.data))
            self.data.iloc[:,0] = clean(self.data.iloc[:,0])       
    
    def __getitem__(self, idx):

        if not self.EthnoSpec:
            text, *rest = self.data.iloc[idx]
            ethnicity = 'nope'
        else:
            text, ethnicity, _, *rest = self.data.iloc[idx]
            if self.ethnicity_processing=='internal':
                text = ethnicity + '[SEP]' + text

            elif self.ethnicity_processing == 'external': #TO-DO
                return text, ethnicity, rest
            
        #if rest!=[]:
        #    rest = self.proc_rest(rest)

        return self.data.index[idx], ethnicity, text, rest
#Майнер    
class BertMiner():
    def __init__(self, workspace, name, miner, miner_data, target_name, miner_threshold, 
                 miner_batch_size, classes_to_find, device='cuda', get_data=True):
        
        self.target_name = target_name
        print('Loading miner...')
        try:
            self.miner = torch.load(f'{workspace}/{target_name}/cache/miners/{name}/{miner}/best_model').to(device)
        except:
            self.miner = torch.load(f'{workspace}/{target_name}/cache/miners/{name}/{miner}/best_checkpoint').to(device)
        self.miner.eval()

        self.seen_data = None
        print('1 chunck = 10000 obs')
        if get_data:
            df = pd.concat([chunk for chunk in tqdm.notebook.tqdm(pd.read_csv(miner_data, 
                                                                          chunksize=10000, 
                                                                          index_col=0), 
                                                                          desc='Loading data')])
            df = df[['source_text']]
            self.loader = DataLoader(EthnoHateMinerDataset(df, type='mine'), 
                                 batch_size=miner_batch_size, shuffle=True)
            if self.miner.prompt is not None:
                self.loader.dataset.prompt = self.miner.prompt
                self.loader.dataset.modify()
            print('Data is ready!!!')
            self.ethno_specific = self.loader.dataset.EthnoSpec
            self.batch_size = miner_batch_size
            self.type = 'mining'
            self.classes_to_find = classes_to_find

            columns = ['source_text', target_name]
            if self.loader.dataset.EthnoSpec:
                columns.insert('eth_group_to_code', 2)
            self.mined_data = pd.DataFrame(columns=columns)
            self.mined_data['score_0'] = np.nan
            self.mined_data['score_1'] = np.nan

    def mine(self, save_all_scores=False, path=None, threshold=None):
        if save_all_scores:
            loop = tqdm.notebook.tqdm(self.loader, position=0, leave=True)
            circle = 1 if circle is None else circle
            for data in loop:
                loop.set_description(f"mining...") 
                if not self.loader.dataset.EthnoSpec or self.loader.dataset.ethnicity_processing=='internal':
                    id, ethnicity, text, rest = data 
                else: # TO-DO
                    pass   
        #выбираем наблюдения по порогу уверенности и добавляем в датасет итеративно
                with torch.no_grad():
                    logits, *rest = self.miner(text, rest)
                    probs = F.softmax(logits, dim=-1)
                temp = {'source_text': text,
                        'score_0' :  probs[:,0].cpu().tolist(),
                        'score_1' :  probs[:,1].cpu().tolist(),
                        self.target_name : probs.argmax(dim=1).cpu().tolist()}
                temp = pd.DataFrame(temp, index = id.int().tolist())
                self.mined_data = pd.concat([self.mined_data.iloc[:], temp.iloc[:]])
        else:
            try:
                self.mined_data = pd.read_csv(path)
                cond = self.mined_data[['score_0', 'score_1']].max(1).between(threshold[0], threshold[1])
                self.mined_data = self.mined_data[['source_text', self.target_name]].loc[cond] 
                if self.classes_to_find == 'balanced':
                    zero_class = self.mined_data.loc[self.mined_data[self.target_name]==0]
                    one_class = self.mined_data.loc[self.mined_data[self.target_name]==1]
                    self.mined_data = pd.concat([zero_class.iloc[:], one_class.iloc[:zero_class.shape[0]]])
                elif type(self.classes_to_find) == list:
                    self.mined_data = self.mined_data[self.mined_data[self.target_name].isin(self.classes_to_find)]
                print(f'Shape of mined filtered data is {self.mined_data.shape}')
            except:
                self.mine(save_all_scores=True)
                self.mine( save_all_scores=False, path=None, threshold=threshold)
        #if self.seen_data is not None:
        #    self.seen_data = pd.concat([self.seen_data, self.mined_data])
        #    self.mined_data = pd.DataFrame(columns=self.seen_data.columns)
                           