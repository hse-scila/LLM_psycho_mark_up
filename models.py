import torch
import copy
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification
from torch.nn import functional as F
from losses import Simcse
from sentence_transformers.losses import BatchSemiHardTripletLoss as triplet
from data_utils import EthnoHateDataset
from transformers import get_linear_schedule_with_warmup, AdamW


class BasicBertModel(nn.Module):
    
    #подходит для: 
    #cointegrated/rubert-tiny2
    
    #SberDevices:
    
    #ai-forever/ruBert-base/large
    #ai-forever/ruRoberta-large
    #ai-forever/sbert_large_mt_nlu_ru
    
    #Trained for similar classification problems models:
    
    #IlyaGusev/rubertconv_toxic_clf
    #cointegrated/rubert-tiny-toxicity
    #apanc/russian-inappropriate-messages
    #s-nlp/russian_toxicity_classifier
    #MonoHime/rubert-base-cased-sentiment-new
    #cointegrated/rubert-tiny-sentiment-balanced
    #blanchefort/rubert-base-cased-sentiment-rusentiment
    #Aniemore/rubert-tiny2-russian-emotion-detection
    #cointegrated/rubert-tiny2-cedr-emotion-detection
    #MaxKazak/ruBert-base-russian-emotion-detection
    
    def __init__(self, n_classes, hugging_face_model, config, n_classes_additional=[]):
        super().__init__()
        
        self.masked = config.masking
        self.n_classes = n_classes
        self.classifier_layers = config.classifier_layers
        self.length = config.length
        self.pooling = config.pooling
        self.dropout = config.dropout
        self.normalize = config.normalize
        self.hf_model = hugging_face_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
        self.concatenate = config.concatenate 
        self.triplet_type = config.triplet_type
        self.hook = config.hook
        self.compute_triplet = config.triplet
        self.compute_diversity = config.compute_diversity
        self.diversity_layers = config.diversity_layers
        self.diversity_strategy = config.diversity_strategy
        self.additional_features = config.additional_features
        self.add_tokens = config.add_tokens
        self.prompt = None
        self.masking = config.masking
        self.masking_type = config.masking_type
        self.use_pooler = config.use_pooler
        
        if self.triplet_type == 'simcse':
            self.triplet = Simcse(model=None)  
        else: 
            self.triplet = triplet(config.simcse_temp)
        
        self.activation = {}
        self.device = config.device
        
        if self.pooling in ['MLM', 'soft MLM']:
            self.bert = AutoModelForMaskedLM.from_pretrained(self.hf_model)
            self.bert.embeddings = self.bert.base_model.embeddings
            self.bert.encoder = self.bert.base_model.encoder
        else:
            self.bert = AutoModel.from_pretrained(self.hf_model)
            try:
                self.bert = self.bert.base_model
            except:
                print('no base model')
        try:
            self.last_layer_input_dim = self.bert.embeddings.word_embeddings.embedding_dim
        except:
            self.last_layer_input_dim = self.bert.base_model.embeddings.word_embeddings.embedding_dim
        
        #Если мы хотим конкатенировать финальный аутпут с промежуточным, то нужно зарегистрировать соответствующие хуки
        n = 1
        if self.concatenate is not None:
            for hook in self.concatenate:
                if self.hook is not None:
                    if hook < self.hook:
                        self.bert.encoder.layer[hook].register_forward_hook(self.get_activation(f'{hook} block'))
                        n+=1
                    else:
                        self.concatenate.remove(hook)
                        print(f'Ignoring hook {hook} because it is not less than output hook')
                else:
                    self.bert.encoder.layer[hook].register_forward_hook(self.get_activation(f'{hook} block'))
                    n+=1
                           
        if self.hook is not None:
            self.bert.encoder.layer[self.hook].register_forward_hook(self.get_activation(f'{hook} block')) 
                       
        self.last_layer_input_dim = self.last_layer_input_dim*n
            
        if self.pooling=='[CLS]+mean' or self.pooling == 'prefix+mean':
            self.last_layer_input_dim = self.last_layer_input_dim * 2       
        #Создаем классификатор

        n_out_classes = [self.n_classes] + n_classes_additional
        self.classifiers = []
        for i in range(len(n_out_classes)):
            modules = []
            inp = self.last_layer_input_dim+self.additional_features
            for out in self.classifier_layers:
                modules.append(nn.Linear(inp, out))
                if config.non_linear:
                    modules.append(nn.GELU())
                modules.append(nn.Dropout(p=self.dropout))
                inp = out
            modules.append(nn.Linear(inp, n_out_classes[i]))
            self.classifiers.append(nn.Sequential(*modules))
        self.classifiers = nn.ModuleList(self.classifiers)
        
        self.embs_to_train = 'all'
        if self.pooling in ['MLM', 'soft MLM']:
            try:
                self.yes = self.tokenizer.vocab['да']
                self.no = self.tokenizer.vocab['нет']
            except:
                self.yes = self.tokenizer(['да']).input_ids[0][1]
                self.no = self.tokenizer(['нет']).input_ids[0][1]
            if self.pooling == 'MLM':
                self.prompt = config.question.replace('[MASK]', self.tokenizer.mask_token)
            else:
                self.tokenizer.add_tokens([f'mtk{i}' for i in range(self.add_tokens)])
                self.bert.base_model.resize_token_embeddings(len(self.tokenizer))
                self.prompt = ' '.join([f'mtk{i}' for i in range(self.add_tokens)]) + '"[TEXT]" ? [MASK]'
                self.prompt = self.prompt.replace('[MASK]', self.tokenizer.mask_token)
                if config.strategy == 'classifier_only':
                    self.embs_to_train = self.add_tokens
        if self.pooling in ['prefix','prefix+mean']:
            self.prompt = config.prefix
            self.prompt = self.prompt.replace('[SEP]', self.tokenizer.sep_token)
            self.prefix_n = self.tokenizer([self.prompt], return_tensors='pt')['input_ids'][0]
            self.prefix_n = len(self.prefix_n)
        if self.pooling == 'prompt-based':
            self.prompt = config.prompt.replace('[MASK]', self.tokenizer.mask_token)
        if self.pooling in ['soft-prompt', 'soft-prompt+mean']:
            self.tokenizer.add_tokens([f'mtk{i}' for i in range(self.add_tokens)])
            self.bert.base_model.resize_token_embeddings(len(self.tokenizer))
            self.prompt = ' '.join([f'mtk{i}' for i in range(self.add_tokens)]) + ' [TEXT]'
            if config.strategy == 'classifier_only':
                self.embs_to_train = self.add_tokens
        
    def forward(self, input, rest, classifier = 0):
        #токенизируем
        tokens = self.tokenizer(input, return_tensors='pt', padding=True, 
                                truncation=True, max_length=self.length).to(self.device)
        #пропускаем через bert
        emb = self.bert(**{k: v for k, v in tokens.items()}, output_attentions=self.compute_diversity)
        if self.pooling in ['MLM', 'soft MLM']:
            return emb[0][:, -2, [self.no, self.yes]], None, torch.tensor(0)
        if self.compute_diversity:
            attentions = emb.attentions
        if self.hook is not None:
            emb = self.activation[f'{self.hook} block'][0]
        else:
            pooler_output = 0
            if self.use_pooler:
                pooler_output = emb.pooler_output
            emb = emb.last_hidden_state
            
        #конкатенируем с промежуточными представлениями, если нужно 

        triplet_embs = None,
        if self.compute_triplet:
            triplet_embs = emb[:, 0, :] if self.pooling != 'prompt-based' else emb[:, -2, :]

        emb = self.pool(emb, tokens, pooler_output)
        #если хотим, нормализуем
        if self.normalize:
            emb = F.normalize(emb)
        if self.additional_features>0:
            emb = torch.cat([emb, rest.to(self.device)], dim=1)
        
        diversity = torch.tensor(0).float()
        if self.compute_diversity:
            if len(self.diversity_layers)==1:
                attentions = attentions[self.diversity_layers[0]].unsqueeze(0)
            else:
                attentions = torch.stack(attentions[self.diversity_layers[0]:self.diversity_layers[1]])
            diversity = self.diversity(attentions)
        #получаем логиты 

        return self.classifiers[classifier](emb), emb, diversity

    def pool(self, emb, tokens, pooler_output=None):
        if self.pooling == 'mean':
            input_mask_expanded = tokens.attention_mask.unsqueeze(-1).expand(emb.size()).float()
            emb = torch.sum(emb * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            emb = emb/sum_mask
        elif self.pooling == 'CLS':
            if self.use_pooler:
                emb = pooler_output
            else:
                emb = emb[:, 0, :]
        elif self.pooling == '[CLS]+mean':
            temp = emb[:,1:,:]
            input_mask_expanded = tokens.attention_mask.unsqueeze(-1)[:,1:,:].expand(temp.size()).float()
            temp = torch.sum(temp * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            temp = temp/sum_mask
            if self.use_pooler:
                emb = pooler_output
            else:
                emb = emb[:, 0, :]
            emb = torch.cat([emb,  temp], dim=1)
        elif self.pooling == 'prompt-based':
            emb = emb[:, -2, :]
        elif self.pooling == 'prefix':
            emb = emb[:, :self.prefix_n, :].mean(axis=1)
        elif self.pooling == 'prefix+mean':
            temp = emb[:,self.prefix_n:,:]
            input_mask_expanded = tokens.attention_mask.unsqueeze(-1)[:,self.prefix_n:,:].expand(temp.size()).float()
            temp = torch.sum(temp * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            temp = temp/sum_mask
            emb = torch.cat([emb[:, :self.prefix_n, :].mean(axis=1),  temp], dim=1)
        elif self.pooling == 'soft-prompt':
            emb = emb[:,1:self.add_tokens+1,:].mean(axis=1)
        elif self.pooling == 'soft-prompt+mean':
            temp = emb[:,self.add_tokens+1:,:]
            input_mask_expanded = tokens.attention_mask.unsqueeze(-1)[:,self.add_tokens+1:,:].expand(temp.size()).float()
            temp = torch.sum(temp * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            temp = temp/sum_mask
            emb = torch.cat([emb[:,1:self.add_tokens+1,:].mean(axis=1), ], dim=1)
        return emb

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output
        return hook
    
    def diversity(self, attention):
        o, a, b, c, d = attention.shape
        if self.diversity_strategy == 'equal':         
            attention = attention.reshape(o*a,b,c*d)
            attention = (attention - attention.mean(dim=2, keepdim=True))/attention.std(dim=2, keepdim=True)
            attention = torch.bmm(attention, attention.permute(0,2,1))/(c*d-1) - torch.eye(b).to('cuda')
            diversity = (attention**2).mean()
        else:
            if self.diversity_strategy == 'increasing':
                weights = torch.linspace(0.1,1,o)
            elif self.diversity_strategy == 'decreasing':
                weights = torch.linspace(1,0.1,o)
            diversity = 0
            for i in range(o):
                at = attention[i,:,:,:].reshape(a,b,c*d)
                at = (at - at.mean(dim=2, keepdim=True))/at.std(dim=2, keepdim=True)
                at = torch.bmm(at, at.permute(0,2,1))/(c*d-1) - torch.eye(b).to('cuda')
                at = (at**2).mean()
                diversity += at*weights[i]
        return diversity
    

class ZeroShotBertModel(nn.Module):
    #подходит для: 
    def __init__(self, hugging_face_model, config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(hugging_face_model)
        self.bert = AutoModelForSequenceClassification.from_pretrained(hugging_face_model)
        self.pos_neg = config.nli_pos_neg
        self.prompt = None
        self.device = config.device
        self.embs_to_train = 'all'

    def forward(self, input, rest, no):
        try:
            tokens = self.tokenizer(input[0], input[1], return_tensors='pt', 
                                    padding=True, truncation=True, max_length=512, 
                                    truncation_strategy='only_first').to(self.device)
        except:
            for n,i in enumerate(input[0]):
                if type(i)!=str:
                    print(n, i)
            print('------------------')
            print(input[1])
            assert False
        logits = self.bert(**{k: v for k, v in tokens.items()})[0]
        return logits[:, self.pos_neg], torch.tensor(0).float(), torch.tensor(0).float()
                    