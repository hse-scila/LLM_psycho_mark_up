import torch
import tqdm
import torch.distributed as dist
import string
from itertools import zip_longest
import torch.multiprocessing as mp
from configuration import Config
import traceback
import random
from results_df import  convert_pickle_to_df
from IPython.display import clear_output
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import itertools
from data_utils import prepare_data_single_target, split_train_test, EthnoHateDataset, get_resampler
from data_utils import clean
from augmentations import Augmentizer
from mining import BertMiner
from collections import defaultdict
from models import BasicBertModel, ZeroShotBertModel
import os
import json
from datetime import datetime
import time
from transformers import get_linear_schedule_with_warmup, AdamW

class EthnoHateTrainer():
    def __init__(self, model, train_loader, val_loader, configs, 
                 additional_train_loaders=None, additional_val_loaders=None):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.additional_train_loaders = additional_train_loaders
        self.additional_val_loaders = additional_val_loaders

        self.multiple_tasks = configs.multiple_tasks
        self.n_tasks = configs.n_tasks
        self.tasks = ['Ethno'] + configs.tasks 

        self.strategy = configs.strategy
        print(f'Using {self.strategy} strategy')
        if self.strategy == 'full':
            params_to_train = self.model.parameters()
        elif self.strategy == 'half':
            n = len(self.model.bert.encoder.layer)
            self.half = n//2 + n%2 
            params_to_train = list(self.model.bert.encoder.layer[-self.half:].parameters())
            params_to_train = params_to_train + list(self.model.classifiers.parameters())
        elif self.strategy == 'last':
            params_to_train = list(self.model.bert.encoder.layer[-1].parameters())
            params_to_train = params_to_train + list(self.model.classifiers.parameters())
        elif self.strategy == 'classifier_only':
            params_to_train = self.model.classifiers.parameters()
            if isinstance(self.model.embs_to_train, int):    
                params_to_train = list(params_to_train) + list(self.model.bert.embeddings.parameters())
        elif self.strategy!='inference_only':
            params_to_train = list(self.model.bert.encoder.layer[self.strategy[0]:self.strategy[1]].parameters())
            params_to_train = params_to_train + list(self.model.classifiers.parameters())
        
        self.n_epochs = configs.n_epochs

        if configs.opt in ['AdamW', 1] and self.n_epochs>0:
            self.opt = AdamW(params_to_train, lr=configs.lr)
        elif configs.opt in ['Adam', 2] and self.n_epochs>0:
            self.opt = torch.optim.Adam(params_to_train, lr=configs.lr)

        if configs.parallel:
            self.opt = torch.nn.parallel.DistributedOptimizer(self.opt)
        self.configs = configs
        self.scaler = torch.cuda.amp.GradScaler()

        self.scheduler = configs.scheduler
        if self.scheduler is not None and self.n_epochs>0:
            if self.scheduler =='default':
                print('Using default linear warmup scheduler!') 
                self.scheduler = get_linear_schedule_with_warmup(self.opt, 
                                                                 len(self.train_loader)*configs.scheduler_fraction, 
                                                                 len(self.train_loader)*self.n_epochs)                                                            
        
        self.weights=None
        if configs.weights is not None:
            if configs.weights == 'proportional':
                self.weights = (self.train_loader.dataset.data[self.configs.target_name].value_counts()/len(self.train_loader.dataset)).sort_index()
                self.weights = (1-torch.tensor(self.weights)).to(self.configs.device).float()
            elif configs.weights != 'resample':
                self.weights = (1-torch.tensor(configs.weights)).to(self.configs.device).float()
        
        self.compute_cross = configs.cross

        self.compute_triplet = configs.triplet
        self.triplet_weight = configs.triplet_weight
        self.compute_diversity = configs.compute_diversity
        self.diversity_weight = configs.diversity_coef
        self.label_smoothing = configs.label_smoothing

        self.multiple_tasks = configs.multiple_tasks
        self.n_tasks = configs.n_tasks
        self.tasks = ['Ethno'] + configs.tasks 
        self.coefs = configs.coefs

        self.all_losses = [defaultdict(list) for _ in range(self.n_tasks)]
        self.metrics_val  = [pd.DataFrame() for _ in range(self.n_tasks)] 
        self.metrics_train = [pd.DataFrame() for _ in range(self.n_tasks)]

        self.fp16 = configs.fp16

        self.keep = configs.keep
        self.renew_idx = configs.renew_idx
        self.aug_rest = configs.aug_rest
        self.renew_aug = configs.renew_aug
        try:
            self.nli = configs.nli
        except:
            self.nli = False
        self.configs = configs
    
    def compute_loss(self, text, rest, target, task):
        
        logits, embs, loss_diversity = self.model(text, rest, task)
        if self.compute_cross:
            loss_cross = F.cross_entropy(logits, target, weight = self.weights, 
                                         label_smoothing = self.label_smoothing)
        else:
            loss_cross = torch.tensor(0).float()
        if self.compute_triplet: 
            if self.model.triplet_type == 'semihard':
                loss_trip = self.model.triplet.batch_semi_hard_triplet_loss(target, embs)*self.triplet_weight
            elif self.model.triplet_type == 'simcse':
                loss_trip = self.model.triplet(embs, target)
        else:
            loss_trip = torch.tensor(0).float()

        loss = loss_cross + loss_trip*self.triplet_weight
        if self.compute_diversity:
            loss = loss + loss_diversity*self.diversity_weight
        if self.compute_cross:
            self.all_losses[task]['loss_cross'].append(loss_cross.item())
        if self.compute_triplet:
            self.all_losses[task]['loss_triplet'].append(loss_trip.item())
        if self.compute_diversity:
            self.all_losses[task]['loss_diversity'].append(loss_diversity.item())
    
        return loss, logits, loss_cross, loss_trip, loss_diversity
      
    def zero_embs(self):
         if isinstance(self.model.embs_to_train, int):    
            self.model.bert.embeddings.word_embeddings.weight.grad[:-self.model.embs_to_train]=0
    
    def train_one_epoch(self, i):
        self.answers_train = [[] for _ in range(self.n_tasks)] 
        self.target_train = [[] for _ in range(self.n_tasks)]
        self.idx = []
        loops = [tqdm.notebook.tqdm(self.train_loader, position=0, leave=True)]
        if self.multiple_tasks:
            for j, tr in enumerate(self.additional_train_loaders):
                loops.append(tqdm.notebook.tqdm(tr, position=j+1, leave=True))

        loss_cross, loss_trip, loss_diversity = torch.tensor(0).float(), torch.tensor(0).float(),  torch.tensor(0).float()

        for data in zip(*loops):
            if not self.train_loader.dataset.EthnoSpec or self.train_loader.dataset.ethnicity_processing=='internal':
                for no, d in enumerate(data):
                    if d is not None:
                        self.opt.zero_grad()
                        text, target, rest, idx = d   
                        target = target.type(torch.LongTensor).to(self.model.device)
                #autocast to fp16 -> less vram usage
                        if self.fp16:
                            with torch.cuda.amp.autocast():
                                loss, logits, loss_cross, loss_trip, loss_diversity = self.compute_loss(text, rest, target, no)
                                loss = loss*self.coefs[no]
                            self.scaler.scale(loss).backward()
                            self.zero_embs()
                            self.scaler.step(self.opt)
                            self.scaler.update()
                        else:
                            loss, logits, loss_cross, loss_trip, loss_diversity = self.compute_loss(text, rest, target, no)
                            loss = loss*self.coefs[no]
                            loss.backward()
                            self.zero_embs()
                            self.opt.step()
                        probs = F.softmax(logits, dim=-1).detach()
                        self.answers_train[no].append(torch.argmax(probs, dim=-1).cpu())
                        self.target_train[no].append(target.cpu())
                        if no==0:
                            self.idx.extend(idx)     
                        loops[no].set_description(f"{self.tasks[no]}: epoch {i+1} of {self.n_epochs}: cros={round(loss_cross.item(),3)}; trip={round(loss_trip.item(),3)}; div={round(loss_diversity.item(),3)}") 
                if self.scheduler is not None:
                    self.scheduler.step()           
                    
            else: #TO-DO
                text, ethnicity, target, rest = data

        for j in range(self.n_tasks):
            self.answers_train[j] = torch.cat(self.answers_train[j])
            self.target_train[j] = torch.cat(self.target_train[j])

        if self.keep is not None:
            cond = self.answers_train[0] == self.target_train[0]
            if self.keep == 'wrong': 
                cond = ~cond
            cond = torch.tensor(self.idx)[cond]
            if not self.renew_idx:
                self.train_loader.idx_to_mask = torch.cat([self.train_loader.idx_to_mask, cond])
                self.train_loader.idx_to_mask = torch.unique(self.train_loader.idx_to_mask)
            else:
                self.train_loader.idx_to_mask = cond

            if self.aug_rest:
                if not self.renew_aug:
                    self.train_loader.idx_to_augment = torch.cat([self.train_loader.idx_to_augment, cond])
                    self.train_loader.idx_to_augment = torch.unique(self.train_loader.idx_to_augment)
                else:
                    self.train_loader.idx_to_augment = cond
   
    def eval_one_epoch(self, i):
        self.answers_val = [[] for _ in range(self.n_tasks)]
        loops = [tqdm.notebook.tqdm(self.val_loader, position=0, leave=True)]
        if self.multiple_tasks:
            for j, v in enumerate(self.additional_val_loaders):
                loops.append(tqdm.notebook.tqdm(v, position=j+1, leave=True))
        for data in zip_longest(*loops):
            for no, d in enumerate(data[self.configs.pass_etno:]):
                if d is not None:
                    if not self.val_loader.dataset.EthnoSpec or self.val_loader.dataset.ethnicity_processing=='internal':
                        text, _, rest, idx = d
                        if self.fp16:
                            with torch.cuda.amp.autocast():
                                logits, _, _ = self.model(text, rest, no)
                        else:
                            logits, _, _ = self.model(text, rest, no)
                    else: #TO-DO
                        text, ethnicity, _, rest = data
                    probs = F.softmax(logits, dim=-1)
                    if not self.nli:
                        self.answers_val[no].append(torch.argmax(probs, dim=-1).cpu())
                    else:
                        self.answers_val[no].append(probs[:,1].detach().cpu())
                    loops[no].set_description(f"{self.tasks[no]}: epoch {i+1} of {self.n_epochs}. Validating...") 

        for j in range(self.n_tasks-int(self.configs.pass_etno)):
            self.answers_val[j] = torch.cat(self.answers_val[j])
    
    def train(self, checkpointing=False, path=None):  
        self.model.to(self.model.device)  
        if self.n_epochs==0:
            with torch.inference_mode():
                self.eval_one_epoch(0) 
                trues = self.val_loader.dataset.data.iloc[:,1].values.astype(int) 
                self.answers_val[0] = self.answers_val[0].reshape(len(trues),
                len(self.val_loader.dataset.nli_data)//len(trues)).argmax(1)
            accum = 0
            for n,p in enumerate(self.configs.labels):
                cond = torch.isin(self.answers_val[0],
                                  torch.tensor([*range(accum, accum+len(p))]))
                self.answers_val[0][cond] = n
                accum += len(p)
            self.metrics_val[0] = plot_metrics(trues, 
                             self.answers_val[0],
                             self.metrics_val[0], 
                            f'f1 scores + accuracy validation set',
                            'zero shot')
        for i in range(self.n_epochs):
        
            self.model.train()
            if self.strategy != 'full':
                self.model.bert.eval()
                if self.strategy == 'half':    
                    self.model.bert.encoder.layer[-self.half:].train()
                elif self.strategy == 'last':
                    self.model.bert.encoder.layer[-1].train()
                elif isinstance(self.strategy, list):    
                    self.model.bert.encoder.layer[self.strategy[0]:self.strategy[1]].train()
                elif isinstance(self.model.embs_to_train, int):     
                    self.model.bert.embeddings.train()
                               
            self.train_one_epoch(i)
       
            self.model.eval()
            self.eval_one_epoch(i)
        
            clear_output(True)
            for j in range(self.n_tasks):
                plot_losses(self.all_losses[j], self.tasks[j])
            
            for j in range(self.n_tasks):

                self.metrics_train[j] = plot_metrics(self.target_train[j].numpy().astype(int), 
                                                  self.answers_train[j].numpy().astype(int), 
                                                  self.metrics_train[j],
                                                  f'f1 scores + accuracy training set',
                                                  self.tasks[j])
                #multiple tasks overhead !!!!                               
                if j==0:
                    self.train_loader.dataset.epoch += 1
                    if self.nli:
                        self.answers_val[j] = self.answers_val[j].reshape(len(self.val_loader.dataset.data),
                        len(self.val_loader.dataset.nli_data)//len(self.val_loader.dataset.data)).argmax(1)
                        accum = 0
                        for n,p in enumerate(self.configs.labels):
                            cond = torch.isin(self.answers_val[j], 
                                   torch.tensor([*range(accum, accum+1)]))
                            self.answers_val[j][cond] = n
                            accum += 1
                    target = self.val_loader.dataset.data.iloc[:,1].values
                else:
                    self.additional_train_loaders[j-1].dataset.epoch+=1
                    target = self.additional_val_loaders[j-1].dataset.data.iloc[:,1].values
                    
                self.metrics_val[j] = plot_metrics(target.astype(int), 
                                                 self.answers_val[j].numpy().astype(int), 
                                                 self.metrics_val[j], 
                                                 f'f1 scores + accuracy validation set',
                                                 self.tasks[j])
            if checkpointing:
                if self.metrics_val[0].T.reset_index()['macro avg'].argmax()==i:
                    print('New best checkpoint !!!')
                    torch.save(self.model, f'{path}/best_checkpoint')
                    print(f'Model saved at {path} !')
        for j in range(self.n_tasks):
            self.metrics_train[j] = self.metrics_train[j].T.reset_index(), 
            self.metrics_val[j] = self.metrics_val[j].T.reset_index()

        if not self.multiple_tasks:
            self.metrics_train, self.metrics_val = self.metrics_train[0], self.metrics_val[0]
        
        return self.metrics_train, self.metrics_val
   
def plot_metrics(trues, answers, metrics, title, task):
    
    report_train = classification_report(trues, answers, output_dict=True)
    report_train = pd.DataFrame(report_train)
    #metrics_train = metrics_train.append(report_train.iloc[2,:]) deprecated in new pandas
    metrics = pd.concat([metrics.loc[:], report_train.iloc[2,:]], axis=1)
    
    sns.lineplot(metrics.T.reset_index().iloc[:,1:])
            
    plt.title(f'{task}: {title}')
    plt.xlabel('epoch')
    plt.show()
    return metrics

def plot_losses(all_losses, task):
    df = pd.DataFrame(all_losses)
    df['loss_cross'] = df['loss_cross'].rolling(100, min_periods=1).mean()
    df.plot()
    
    plt.title(f'{task}: train losses')
    plt.show()

def mine(workspace, name, miner_name, miner_data, target_name, miner_circles, miner_threshold, 
        miner_batch_size, classes_to_find=None, train=True, leave_data=True, leave_model=True,
        masking = None, get_data=True, append_only=None, save_all_scores = False):
    
    if classes_to_find == 'balanced':
        add = 'balanced'
    elif classes_to_find is not None:
        add = f'{classes_to_find}_classes'
    else:
        add = 'all'
    
    miner = BertMiner(workspace, name, miner_name, miner_data, target_name, miner_threshold, 
                 miner_batch_size, classes_to_find, get_data = get_data)
    
    path = f'{workspace}/{target_name}/cache/miners/{name}/{miner_name}/threshold_{miner_threshold}/prev_data_{leave_data}_prev_model_{leave_model}/{add}'
    os.makedirs(path, exist_ok=True)
    os.makedirs(f'{path}/{masking}_{append_only}', exist_ok=True)
    os.makedirs(f'{workspace}/{target_name}/cache/miners/{name}/{miner_name}/{add}_{miner_threshold}', exist_ok=True)
    print('Initializing miner...')
    log = open(f'{path}/log.txt', "w+")
    log.close()
    test_data = pd.read_csv(f'{workspace}/{target_name}/cache/val/data_no_augs.csv', index_col=0)
    train_data = pd.read_csv(f'{workspace}/{target_name}/cache/train/data_no_augs.csv', index_col=0)
    if not leave_data:
        miner.seen_data = train_data
    else:
        miner.mined_data = train_data
    'Starting mining...'
    best_metric = 0
    for i in range(miner_circles):
        try:
            miner.mined_data = pd.read_csv(f'{workspace}/{target_name}/cache/miners/{name}/{miner_name}/{add}_{miner_threshold}/data_{i+1}_circle.csv', index_col=0)
            if append_only is not None:
                miner.mined_data = miner.mined_data.loc[miner.mined_data.iloc[:,1].isin(append_only)]
            print('Using cached mined data!')
        except:
            miner.mine(i+1, save_all_scores=save_all_scores)
            if not save_all_scores:
                miner.mined_data.to_csv(f'{workspace}/{target_name}/cache/miners/{name}/{miner_name}/{add}_{miner_threshold}/data_{i+1}_circle.csv')
            else:
                miner.mined_data.to_csv(f'{workspace}/{target_name}/cache/miners/{name}/{miner_name}/all_scores.csv')      
        if (train or miner_circles>1) and not save_all_scores:
            with open(f'{workspace}/{target_name}/cache/miners/{name}/{miner_name}/best_configs', 'rb') as f:
                configs = pickle.load(f)
                configs.cross = True
            if masking is not None:
                configs.masking = masking[0]
                configs.masking_warm_up = masking[1]
            print(train_data)
            print(miner.mined_data)
            if not leave_model:
                train_data = pd.concat([train_data.iloc[:], miner.mined_data.iloc[:]])
            else:
                train_data = miner.mined_data
            print(train_data)
            train_loader = DataLoader(EthnoHateDataset(train_data, configs, type='train'), 
                batch_size=configs.batch_size, shuffle=True)
            val_loader = DataLoader(EthnoHateDataset(test_data, configs, 
                type='test'), batch_size=configs.batch_size, shuffle=False)
                
            if not leave_model:
                model = BasicBertModel(train_loader.dataset.n_classes, miner.miner.hf_model, configs)
                trainer = EthnoHateTrainer(model, train_loader, val_loader, configs)
                del miner.miner
            else:
                trainer = EthnoHateTrainer(miner.miner, train_loader, val_loader, configs) 
            trainer.train_loader.dataset.mask = model.tokenizer.mask_token
            trainer.train_loader.dataset.tokenizer = model.tokenizer
            metrics_train, metrics_val = trainer.train(True, f'{path}/{masking}_{append_only}')
            if metrics_val['macro avg'].max()>best_metric:
                print(f'New best model !!!')
                best_metric = metrics_val['macro avg'].max()
                if os.path.exists(f'{path}/{masking}_{append_only}/best_model'):
                    os.remove(f'{path}/{masking}_{append_only}/best_model')
                os.rename(f'{path}/{masking}_{append_only}/best_checkpoint', f'{path}/{masking}_{append_only}/best_model')
                with open(f'{path}/{masking}_{append_only}/best_configs', 'wb') as handle:
                    pickle.dump(configs, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                os.remove(f'{path}/{masking}_{append_only}/best_checkpoint')
            del trainer
            if not leave_model:
                del model
            metrics_val.to_csv(f'{path}/{masking}_{append_only}/results_{i+1}.csv')
            #miner.miner = torch.load(f'{path}/best_model')
            mined_data = miner.mined_data
        else:
            return miner.mined_data
    return mined_data, metrics_train, metrics_val

def augment(data, configs, tpe='train'):
    augmentizer = Augmentizer(data, configs, tpe=tpe, local=configs.local) 
    data, additional_features = augmentizer.augment()
    return data, additional_features

def find_or_create_cache(configs, data=None):
    try:
        coding=None
        train_data = pd.read_csv(f'{configs.workspace}/{configs.target_name}/cache/train/data_no_augs.csv', index_col=0)
        val_data = pd.read_csv(f'{configs.workspace}/{configs.target_name}/cache/val/data_no_augs.csv', index_col=0)
        test_data = None
        print('Using cached data...')
    except:
        print("Can't find cached data! Creating data...")
        if data is None:
            data, coding = prepare_data_single_target(configs)
        else:
            data, coding = pd.read_csv(data, index_col=0), None
        print(configs.clean)
        if configs.clean:
            print('Cleaning data')
            data.iloc[:,0] = clean(data.iloc[:,0]) 
        train_data, test_data = split_train_test(data, configs, split=configs.test_size)
        val_data, test_data = split_train_test(test_data, configs, split=configs.val_size)
        train_data.to_csv(f'{configs.workspace}/{configs.target_name}/cache/train/data_no_augs.csv')
        val_data.to_csv(f'{configs.workspace}/{configs.target_name}/cache/val/data_no_augs.csv')
        test_data.to_csv(f'{configs.workspace}/{configs.target_name}/cache/test/data_no_augs.csv')
    return train_data, val_data, test_data, coding

def create_and_fit(model_name, configs, save = False, 
                 save_as_miner=False, best_metric=None, augs_only=False, data=None):
    #look in cache or create
    os.makedirs(f'{configs.workspace}/{configs.target_name}/cache/train', exist_ok=True)
    os.makedirs(f'{configs.workspace}/{configs.target_name}/cache/test', exist_ok=True)
    os.makedirs(f'{configs.workspace}/{configs.target_name}/cache/val', exist_ok=True)
    os.makedirs(f'{configs.workspace}/{configs.target_name}/{configs.name}/{model_name.split("/")[-1]}/models', exist_ok=True)
    if configs.parallel:
        print('Training in parallel...')
        dist.init_process_group(backend="gloo", init_method="tcp://localhost:123456", rank=0, world_size=3)
    train_data, test_data, _, coding = find_or_create_cache(configs,  data=data)
    if configs.augment:
        print("Augmenting data...")
        train_data, configs.additional_features = augment(train_data, configs, tpe='train')
        test_data, configs.additional_features = augment(test_data, configs, tpe='val')  
        if augs_only:
            return 'OK', 'OK', 'OK'

    #resampling
    train_dataset = EthnoHateDataset(train_data, configs, type='train')
    if configs.weights == 'resample':
        if not configs.parallel:
            print('Getting resampler...')
            if not configs.nli:
                sampler = get_resampler(train_data.iloc[:,1])
            else:
                if configs.nli_resample == 'hypo':
                    sampler = get_resampler(train_dataset.nli_data.iloc[:,2])
                elif configs.nli_resample == 'class':
                    #print(pd.Series(np.ones(len(train_dataset.nli_data))[::2]-1))
                    temp = np.ones(len(train_dataset.nli_data))
                    temp[::2] = temp[::2]-1
                    sampler = get_resampler(pd.Series(temp,dtype=int),
                    train_dataset.data.iloc[:,1])
                elif configs.nli_resample == 'hypo+class':
                    sampler = get_resampler(train_dataset.nli_data.iloc[:,2],
                    train_dataset.data.iloc[:,1])
            configs.weights = None
            shuffle = False
        else:
            print('No weighted sampler imlementation for parallel yet!')
            assert False
    else:
        if not configs.parallel:
            sampler = None
            shuffle = True
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            shuffle = False
    #ataloaers (torch style)
    print('Getting loaders...')
    #multi task overhead!
    train_loader = DataLoader(train_dataset, 
                   batch_size=configs.batch_size, shuffle=shuffle, sampler=sampler)
    val_loader = DataLoader(EthnoHateDataset(test_data, configs, type='test'), 
                   batch_size=configs.batch_size, shuffle=False)
    tr_loaders = None
    val_loaders = None
    n_classes = []
    if configs.multiple_tasks:
        tr_loaders = []
        val_loaders = []
        for path in configs.tasks_data:
            data = pd.read_csv(path, index_col=0)
            if data.iloc[:,1].dtype == 'O':
                data.iloc[:,1] = data.iloc[:,1].astype('category')
                data.iloc[:,1] = data.iloc[:,1].cat.codes
            train, test = split_train_test(data, configs)
            if configs.pass_etno:
                train = pd.concat([train.iloc[:],test.iloc[:]])
                test = train
            print(train.head(2))
            if configs.additional_resample:
                print('Getting resampler...')
                sampler = get_resampler(train.iloc[:,1])
                configs.weights = None
                shuffle = False
            else:
                sampler = None
                shuffle = True
            tr_loaders.append(DataLoader(EthnoHateDataset(train, configs, type='train_additional',
            nli_path=path), 
                   batch_size=configs.batch_size, sampler=sampler, shuffle=shuffle))
            val_loaders.append(DataLoader(EthnoHateDataset(test, configs, type='test', nli_path=path), 
                   batch_size=configs.batch_size, shuffle=False))
            n_classes.append(tr_loaders[-1].dataset.n_classes)
    #training    
    print('Initializing model...')
    if not configs.nli:
        model = BasicBertModel(train_loader.dataset.n_classes, model_name, configs, n_classes)
    else:
        model = ZeroShotBertModel(model_name, configs)
    train_loader.dataset.tokenizer = model.tokenizer
    train_loader.dataset.sep_token = model.tokenizer.sep_token
    val_loader.dataset.sep_token = model.tokenizer.sep_token
    if configs.parallel:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0,1,2])
    #multi task overhead!
    train_loader.dataset.mask = model.tokenizer.mask_token
    if configs.multiple_tasks:
        for i in range(len(tr_loaders)):
            tr_loaders[i].dataset.mask = model.tokenizer.mask_token

    if model.prompt is not None:
        print(f'Using {model.prompt} prompt for {model.pooling} pooling')
        train_loader.dataset.prompt = model.prompt
        val_loader.dataset.prompt = model.prompt
        if configs.multiple_tasks:
            for i in range(len(tr_loaders)):
                tr_loaders[i].dataset.prompt = model.prompt
                val_loaders[i].dataset.prompt = model.prompt
                tr_loaders[i].dataset.modify_text()
                val_loaders[i].dataset.modify_text()      
        train_loader.dataset.modify_text()
        val_loader.dataset.modify_text()
        
    trainer = EthnoHateTrainer(model, train_loader, val_loader, configs, tr_loaders, val_loaders)
    print('Training...')
    checkpointing = False
    model_path = None
    if save or save_as_miner or best_metric is not None:
        checkpointing = True
        if save_as_miner:
            model_path = f'{configs.workspace}/{configs.target_name}/cache/miners/{configs.name}/{model_name.split("/")[-1]}'
        elif save:
            model_path = f'{configs.workspace}/{configs.target_name}/{configs.name}/{model_name.split("/")[-1]}'
        else:
            model_path = f'{configs.workspace}/{configs.target_name}/{configs.name}/{model_name.split("/")[-1]}/models'
        os.makedirs(model_path, exist_ok=True)
    metrics_train, metrics_val = trainer.train(checkpointing, model_path)
    if best_metric is not None:
        metrics_val = metrics_val[0] if isinstance(metrics_val, list) else metrics_val
        print(metrics_val['macro avg'].max())
        if metrics_val['macro avg'].max() > best_metric:
            print(f'New best model !!!')
            #print(model_path)
            best_metric = metrics_val['macro avg'].max()
            if os.path.exists(f'{model_path}/best_model'):
                os.remove(f'{model_path}/best_model')
            if configs.n_epochs>0:
                os.rename(f'{model_path}/best_checkpoint', f'{model_path}/best_model')
            else:
                torch.save(model, f'{model_path}/best_model')
            with open(f'{model_path}/best_configs', 'wb') as handle:
                pickle.dump(configs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            if os.path.exists(f'{model_path}/best_checkpoint'):
                os.remove(f'{model_path}/best_checkpoint')

    return metrics_train, metrics_val, best_metric
         
def grid_search(model_name, target_name, names_values_dictionary, workspace, name,  
                rules=None, save_every=10, detailed_log = True, save_as_miner = False, 
                augs_only=False,  data=None):
    
    model = model_name.split('/')[-1]
    os.makedirs(f'{workspace}/{target_name}/cache/train', exist_ok=True)
    os.makedirs(f'{workspace}/{target_name}/cache/test', exist_ok=True)
    os.makedirs(f'{workspace}/{target_name}/cache/val', exist_ok=True)
    os.makedirs(f'{workspace}/{target_name}/{name}/{model}/models', exist_ok=True)
    if not augs_only:
        os.makedirs(f'{workspace}/{target_name}/cache/miners/{name}/{model_name.split("/")[-1]}',
                 exist_ok=True)
        os.system(f'cp analysis.ipynb  {workspace}/{target_name}/{name}/{model}/analysis.ipynb')
        os.system(f'cp results_df.py  {workspace}/{target_name}/{name}/{model}/results_df.py')
    with open(f'{workspace}/{target_name}/{name}/{model}/experiments_series_configs.json', 'w',  
              encoding="utf8") as handle:
        json.dump(names_values_dictionary, handle)
    with open(f'{workspace}/{target_name}/{name}/{model}/experiments_series_rules.json', 'w',  
              encoding="utf8") as handle:
        json.dump(names_values_dictionary, handle)

    experiments_results = dict()
    names = list(names_values_dictionary.keys())
    prod = itertools.product(*names_values_dictionary.values())
    exp = 0

    log_path = f'{workspace}/{target_name}/{name}/{model}/log.txt'
    detailed_log_path = f'{workspace}/{target_name}/{name}/{model}/detailed_log.txt'
    results_path = f'{workspace}/{target_name}/{name}/{model}/results.pickle'
    print(f'Logging will be stored at {log_path}')
    print(f'Results will be stored at {results_path}')
    last_exp = 0

    if detailed_log:
       print(f'Detailed logging will be stored at {detailed_log_path}')
       log = open(detailed_log_path,"w+")
       log.close() 
    
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            lines = f.readlines()
            try:
            #if True:
                with open(results_path, 'rb') as f:
                    experiments_results = pickle.load(f)
                if lines[-1].split(' ')[0] != 'Error:':
                    best_metric = float(lines[-1].split(' ')[-1])
                    last_exp = int(lines[-3].split(' ')[-1])
                else:
                    best_metric = float(lines[-2].split(' ')[-1])
                    last_exp = int(lines[-4].split(' ')[-1])
            #if False:
            except:
                print("Can't find results.pickle file! starting experiments from the begining...")
                last_exp=0
                best_metric=0
                    
    else:
        log = open(log_path, "w+")
        log.write(f'{str(datetime.fromtimestamp(time.time()).strftime("%A, %B %d, %Y %I:%M:%S"))} start time')
        log.write("\n")
        log.close()
        best_metric = 0
        
    for params in prod:
        init = dict()
        for j,i in enumerate(params): 
            init[names[j]] = i
            
        stop = False
        try:
            configs = Config(target_name, workspace, name, **init)
        except:
            stop=True  
        if rules is not None:
            for r in rules:
                if np.all([init[k] in r[k] for k in r.keys()]):
                    stop = True
                    break
        if stop:
            continue
        exp = exp + 1        
        if exp<last_exp:
            continue
                
        log = open(log_path, "a")
        log.write(f'{str(datetime.fromtimestamp(time.time()).strftime("%A, %B %d, %Y %I:%M:%S"))} exp {exp}')
        log.write("\n")
        log.write(json.dumps(init))
        log.write("\n")
        log.write(f"best metric (macro avg) = {best_metric}")
        log.write("\n")
        log.close()
        
        print(f'Starting experiment no {exp}...')
        try:
            set_seed(228)
            metrics_train, metrics_val, best_metric = create_and_fit(model_name, configs, 
                                                               best_metric=best_metric,
                                                               save_as_miner=save_as_miner,
                                                               augs_only=augs_only,
                                                               data=data)
        except Exception as e:
        #if False:
            log = open(log_path, "a")
            log.write(f'Error: {str(e)}')
            log.write("\n")
            if detailed_log:
                log = open(detailed_log_path, "a")
                log.write(f'Exp {exp}')
                log.write("\n")
                log.write(traceback.format_exc())
                log.write("\n")
                log.close()
            #continue    
            break
            
        if not augs_only:
            experiments_results[exp] = {'parameters' : init, 'train_results' : metrics_train, 
                                    'val_results':metrics_val}
            if exp % save_every == 0:
                with open(results_path, 'wb') as handle:
                    pickle.dump(experiments_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            torch.cuda.empty_cache()

    if not augs_only:
        with open(results_path, 'wb') as handle:
            pickle.dump(experiments_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        convert_pickle_to_df(results_path, save=True)

def error_analysis(model_path, val_data_path, batch_size=24, device='cuda'):
    
    model = torch.load(f'{model_path}/best_model').to(device)
    data = pd.read_csv(val_data_path, index_col=0)
    answers_val = []
    with open(f'{model_path}/best_configs', 'rb') as handle:
        configs = pickle.load(handle)
    
    val_loader =  DataLoader(EthnoHateDataset(data, configs=configs, type='test'), 
                             batch_size=batch_size, shuffle=False)  
    try:
        print(val_loader.dataset.labels)
    except:
        configs.nli=False  
    with torch.no_grad():
        for d in tqdm.notebook.tqdm(val_loader):
            text, _, _, _ = d
            logits, _, _ = model(text, None, 0)
            probs = F.softmax(logits, dim=-1)
            if not configs.nli:
                answers = torch.argmax(probs, dim=-1).cpu()
            else:
                answers = probs[:,1].cpu()
            answers_val.append(answers)
    
    answers_val = torch.cat(answers_val).numpy()
    if configs.nli:
        val_loader.dataset.nli_data['answers'] = answers_val
        nli_data = val_loader.dataset.nli_data
        #answers_val = nli_data.answers.values.reshape(len(data),len(nli_data)//len(data)).argmax(1)//len(configs.labels[0])
        trues = val_loader.dataset.data.iloc[:,1].to_numpy()
        answers_val = nli_data.answers.values.reshape(len(val_loader.dataset.data),
                    len(val_loader.dataset.nli_data)//len(val_loader.dataset.data)).argmax(1)
        print(answers_val.shape)
        accum = 0
        for n,p in enumerate(configs.labels):
            cond = torch.isin(torch.tensor(answers_val), torch.tensor([*range(accum, accum+1)]))
            answers_val[cond] = n
            accum += 1
    else:
        trues = val_loader.dataset.data.iloc[:,1].to_numpy()
    return classification_report(trues, answers_val, digits=3), trues, answers_val, data

def get_number_of_models(names_values_dictionary, rules = None, check=False):
    prod = [i for i in itertools.product(*names_values_dictionary.values())]
    names = list(names_values_dictionary.keys())
    exps = 0
    for params in prod:
        init = dict()
        for j,i in enumerate(params):
            init[names[j]] = i
            
        stop = False
        if rules is not None:
            for r in rules:
                if np.all([init[k] in r[k] for k in r.keys()]):
                    stop = True
                    #print('stopping')
                    break
        if not check:
            try:
        #if True:
                configs = Config(None, None, None, **init)
            except:
                #configs = Config(None, None, None, **init)
                stop=True
        else:
            configs = Config(None, None, None, **init)
        if stop:
            continue
        exps+=1
    return exps

def set_seed(seed):
    """ Set all seeds to make results reproducible (deterministic mode).
        When seed is a false-y value or not supplied, disables deterministic mode. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
