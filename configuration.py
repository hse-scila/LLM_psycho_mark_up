class Config:
    def __init__(self, 
                 
                 target_name,
                 workspace,
                 name,
                 
                 nli = False,
                 nli_pos_neg = None,
                 nli_template = None,
                 func = 'mode',
                 nli_labels = None,
                 nli_path = None,
                 nli_resample=None,
                 pass_etno = False,

                 n_tasks = 1,
                 tasks = [],
                 tasks_data = [],
                 coefs = [1],
                 ethno_specific = False, 
                 ethnicity_processing = 'internal',
                 drop = ['unk'],
                 coding = 'auto',
                 test_size = 0.3,
                 val_size = 0.5,
                 split_on_document = False,
                 stratified = None,
                 clean = True, 
                 additional_features = 0, 
                 additional_resample = False,
                 
                 paraphraser=None, 
                 paraphraser_task = '', 
                 FRED = None,
                 FRED_regime = '',
                 FRED_impute = False,
                 FRED_n_mask = None,
                 FRED_fraction = None,
                 FRED_n_to_add = None,
                 FRED_add_random = False,
                 FRED_threshold = None,
                 FRED_sep = '',
                 gpt_restorer = None, 
                 gpt_fraction=None,
                 imputer = None,
                 imputer_masking = None,
                 imputer_masking_type = None,
                 translator = None,
                 translator_lang = None, #en
                 aug_rest = False,
                 original_text_p = 1,
                 aug_warming_up=0,
                 renew_aug = False,
                 preprocessing_translator = None,
                 preprocessing_translator_batch = 1,
                 preprocessing_lang = None,
                 postprocessing_translator = None,
                 repetition_penalty = None,
                 diversity_penalty= None,
                 alpha_penalty = None,
                 top_k = None,
                 temperature = None,
                 grams = None,
                 encoder_grams = None,
                 beams = 1,
                 beam_groups = 1,
                 n_samples = 1,
                 do_sample = False,
                 length_multiplier = None,
                 max_length = None,
                 augmentation_batch = 1,
                 aug_controller = None,
                 aug_controller_regime = '',
                 aug_controller_max = 0,
                 controller_max_diff = 0,
                 etno_replacer = False,
                 etno_replacer_regime = None,
                 etno_replacer_warming = 0,
                 replace_miss_spelled = False,
                 extractor = False,
                 extractor_window = None,
                 check_etnonyms = False,
                 n_safe = 0,
                 add_sep = False,
                 min_length = 0,
                 true_etno_share = 1,

                 toxicity = None, 
                 sentiment=None, 
                 emotion=None,

                 augment_classes = None,
                 only_aug = False,
                 aug_device='cuda',
                 local = False,

                 train_in_foreign = False, 
                 parallel = False,
                 masking = 0,
                 masking_type = None,
                 masking_warm_up = 0,
                 weighting_type = None,
                 reverse_weighting = False,
                 keep = None,
                 renew_idx = False,
                 strategy = 'full',
                 pooler = False,
                 opt = 1,
                 scheduler = None,
                 scheduler_fraction = 0,
                 non_linear = True,
                 fp16 = False,
                 length = 512,
                 pooling = 'mean', 
                 add_tokens = 0,
                 prefix = None, 
                 prompt = None, 
                 question = None, 
                 hook = None,
                 normalize = False,
                 dropout = 0.2,
                 classifier_layers = [256], 
                 concatenate = None, 
                 compute_cross = True,
                
                 triplet = False,
                 triplet_weight=0,
                 triplet_type=None,
                 simcse_temp = None,
                 diversity = False,
                 diversity_coef = 0,
                 diversity_layers = [],
                 diversity_strategy = None,
                 weights = None,
                 label_smoothing = 0,

                 batch_size = 32,
                 lr = 0.00001,
                 n_epochs = 5,
                 
                 device = 'cuda'):
        
        self.workspace = workspace
        self.name = name
        self.opt=opt
        self.parallel = parallel
        self.target_name = target_name
        self.n_tasks = n_tasks
        self.tasks = tasks
        self.tasks_data = tasks_data
        self.ethno_specific = ethno_specific
        self.ethnicity_processing = ethnicity_processing
        self.drop = drop
        self.coding = coding
        self.test_size = test_size
        self.stratified_split = stratified
        self.split_on_document = split_on_document
        self.additional_features = additional_features
        self.clean = clean
        self.prefix = prefix
        self.non_linear = non_linear
        self.use_pooler = pooler
        self.nli = nli
        self.nli_pos_neg = nli_pos_neg
        self.labels = nli_labels
        self.nli_path = nli_path
        self.pass_etno = pass_etno
        self.nli_resample = nli_resample
        
        self.func = func
        self.masking = masking
        self.masking_type = masking_type
        self.reverse_weighting = reverse_weighting
        self.keep = keep
        self.renew_idx = renew_idx
        self.masking_warm_up = masking_warm_up
        self.paraphraser = paraphraser
        self.paraphraser_task = paraphraser_task
        self.length_multiplier = length_multiplier
        self.gpt_restorer = gpt_restorer
        self.gpt_fraction = gpt_fraction
        self.imputer = imputer
        self.imputer_masking = imputer_masking
        self.imputer_masking_type = imputer_masking_type
        self.translator = translator
        self.translator_lang = translator_lang
        self.postprocessing_translator = postprocessing_translator
        self.toxicity = toxicity
        self.sentiment = sentiment
        self.emotion = emotion
        self.prompt = prompt
        self.aug_device = aug_device
        self.encoder_grams = encoder_grams
        self.grams = grams
        self.simcse_temp = simcse_temp
        self.local = local
        self.aug_classes = augment_classes
        self.aug_rest = aug_rest
        self.max_length = max_length
        self.augmentation_batch = augmentation_batch
        self.original_text_p = original_text_p
        self.only_augs = only_aug
        self.n_samples = n_samples
        self.aug_warming_up = aug_warming_up
        self.renew_aug = renew_aug
        self.repetition_penalty = repetition_penalty
        self.diversity_penalty=diversity_penalty
        self.alpha_penalty= alpha_penalty
        self.top_k = top_k
        self.temperature = temperature
        self.preprocessing_translator = preprocessing_translator
        self.preprocessing_lang = preprocessing_lang
        self.preprocessing_translator_batch = preprocessing_translator_batch
        self.train_in_foreign = train_in_foreign
        self.aug_controller = aug_controller
        self.aug_controller_max = aug_controller_max
        self.aug_controller_regime = aug_controller_regime
        self.num_beams = beams
        self.num_beam_groups = beam_groups
        self.do_sample = do_sample
        self.FRED = FRED
        self.FRED_regime = FRED_regime
        self.FRED_impute = FRED_impute
        self.FRED_n_mask = FRED_n_mask
        self.FRED_fraction = FRED_fraction
        self.FRED_n_to_add = FRED_n_to_add
        self.FRED_add_random = FRED_add_random
        self.FRED_threshold = FRED_threshold
        self.FRED_sep = FRED_sep
        self.nli_template = nli_template
        self.controller_max_diff = controller_max_diff
        self.etno_replacer = etno_replacer
        self.etno_replacer_regime = etno_replacer_regime
        self.etno_replacer_warming = etno_replacer_warming
        self.extractor = extractor
        self.extractor_window = extractor_window
        self.replace_miss_spelled = replace_miss_spelled
        self.check_etnonyms = check_etnonyms
        self.n_safe = n_safe
        self.add_sep = add_sep
        self.min_length = min_length
        self.true_etno_share = true_etno_share
        self.additional_resample = additional_resample

        self.cross = compute_cross
        self.scheduler = scheduler
        self.scheduler_fraction = scheduler_fraction
        self.length = length
        self.weighting_type = weighting_type
        self.strategy = strategy
        self.coefs = coefs
        self.fp16 = fp16
        self.hook = hook
        self.pooling = pooling
        self.question = question
        self.add_tokens = add_tokens
        self.normalize = normalize
        self.dropout = dropout
        self.weights = weights
        self.label_smoothing = label_smoothing
        self.classifier_layers = classifier_layers
        self.concatenate = concatenate
        self.triplet_weight = triplet_weight
        self.triplet_type = triplet_type
        self.compute_diversity = diversity
        self.diversity_layers = diversity_layers
        self.diversity_coef = diversity_coef
        self.diversity_strategy = diversity_strategy
        
        self.val_size = val_size
        self.batch_size = batch_size
        self.lr = lr
        self.triplet = triplet
        self.n_epochs = n_epochs
        
        if self.split_on_document and self.stratified_split is not None:
            assert False, 'Error: Cannot stratify when splitting on documents'

        self.multiple_tasks = False
        if self.n_tasks>1:
            self.multiple_tasks = True

        if self.multiple_tasks is not None:
            assert len(self.tasks) == self.n_tasks-1, f"Error: you have to provide {self.n_tasks-1} task names"
            assert len(self.tasks_data) == self.n_tasks-1, f"Error: you have to provide paths to {self.n_tasks-1} datasets for additional tasks!"
            assert len(self.coefs) == self.n_tasks,  f"Error: you have to provide {self.n_tasks-1} task coefs"
        else:
            assert len(self.tasks) == 0, f"Error: Single task, but additional tasks are specified"
            assert len(self.tasks_data) == 0, f"Error: Single task, but additional tasks data are specified"
            assert len(self.coefs) == 1,  f"Error: Single task, but additional coefs are specified"

        self.augment = False
        self.augment_details = False


        if self.translator is not None:
            self.augment = True
            assert self.translator_lang is not None, "Error: You have to specify translator language"
        else: 
            assert self.translator_lang is None, "Error: Translator is None, but language is specified"

        if self.paraphraser is not None:
            self.augment = True
            self.augment_details = True
            if self.augmentation_batch>1:
                self.max_length is not None, "Error: Choose max length for paraphraser"
            else:
                self.length_multiplier is not None, "Error: Choose length multiplier for paraphraser"
        else:
            assert self.paraphraser_task == ''

        if self.FRED is not None:
            self.augment = True
            self.augment_details = True
            assert self.FRED_regime != '', "Error: You have to specify FRED regime"
            assert self.max_length is not None, "Error: You have to specify max length for FRED"
            if self.FRED_impute:
                assert self.FRED_n_mask is not None, "Error: You have to specify FRED n_mask"
                assert self.FRED_n_to_add is not None, "Error: You have to specify FRED n_to_add"
                assert self.FRED_threshold is None, "Error: You have to choose threshold ONLY when not imputing"
            else:
                assert self.FRED_fraction is not None, "Error: You have to specify FRED fraction"
                assert self.FRED_n_mask is None, "Error: You have not to specify FRED n_mask"
                assert self.FRED_n_to_add is None, "Error: You have not to specify FRED n_to_add"
                assert not self.FRED_add_random, "Error: You can add random ONLY for FRED"
                assert self.FRED_regime == '<LM>', "Error: You have to use LM regime if not imputing"
                assert self.FRED_threshold is not None,  "Error: You have to choose threshold for LM regime"
                assert self.FRED_sep == '', 'Error: You cant specify sep for LM FRED'
        else:
            assert self.FRED_fraction is None, "Error: You have not to specify FRED fraction"
            assert self.FRED_n_mask is None, "Error: You have not to specify FRED n_mask"
            assert self.FRED_n_to_add is None, "Error: You have not to specify FRED n_to_add"
            assert not self.FRED_add_random, "Error: You can add random ONLY for FRED"
            assert not self.FRED_impute, "Error: You can impute ONLY for FRED"
            assert self.FRED_regime == '',  "Error: You have to specify FRED regime ONLY when using FRED"
            assert self.FRED_threshold is None, "Error: You have to choose threshold ONLY for FRED"
            assert self.FRED_sep == '', 'Error: You can specify sep ONLY for imputing FRED'

        if self.imputer is not None:
            self.augment = True
            assert self.augmentation_batch == 1, "Error: Batching is not supported for imputer"
            assert self.imputer_masking is not None, "Error: You have to specify imputer masking for imputer"
        else: 
            assert self.imputer_masking is None, "Error: Imputer is None, but imputer masking is specified"
        
        if self.gpt_restorer is not None:
            self.augment_details = True
            self.augment = True
            assert self.length_multiplier is not None, "Error: Please specify length multiplier for gpt restorer"
            assert self.augmentation_batch == 1, "Error: Batching is not supported for gpt_restorer"
            assert self.gpt_fraction is not None, "Error: You have to specify fraction for gpt restorer"
        else:
            assert self.gpt_fraction is None, "Error: GPT restorer is None, but gpt fraction are specified"
        
        if not self.augment:
            #print('here')
            assert self.aug_classes is None, "Error: You have to specify aug classes only in case of augmentation"
            assert not self.aug_rest, "Error: You cant use this regime w/o augmentations"
            assert self.aug_warming_up==0, "Error: You have to use this option only for instance wise augmentation"
            assert not self.renew_aug, "Error: You cant choose this option when not using augmentations"
            assert self.augmentation_batch==1, "Error: You can't specify augmentation batch size when not augmenting"
            assert self.aug_controller is None, "Error: No augs to control"

        if not self.augment_details:
            assert self.n_samples == 1, "Error: You can sample new ONLY when restoring, paraphrasing or summarizing"
            assert self.num_beams == 1, "Error: You can use beam search ONLY when restoring, paraphrasing or summarizing"
            assert self.num_beam_groups == 1, "Error: You can use beam search ONLY when restoring, paraphrasing or summarizing"
            assert self.diversity_penalty is None, "Error: You can use diversity penalty ONLY when restoring, paraphrasing or summarizing"
            assert self.repetition_penalty is None, "Error: You can use repetition penalty ONLY when restoring, paraphrasing or summarizing"
            assert self.grams is None, "Error:  You can specify no_repeat_n_grams ONLY when restoring, paraphrasing or summarizing"
            assert self.encoder_grams is None, "Error:  You can specify no_repeat_n_grams ONLY when restoring, paraphrasing or summarizing"
            assert self.temperature is None,  "Error:  You can specify temperature ONLY when restoring, paraphrasing or summarizing"
            assert not self.do_sample, "Error: You can use multinomial sampling ONLY when restoring, paraphrasing or summarizing"
            assert self.alpha_penalty is None, "Error:  You can use alpha penalty ONLY when restoring, paraphrasing or summarizing"
            assert self.top_k is None, "Error: You have to specify top_k ONLY when restoring, paraphrasing or summarizing"

        if self.alpha_penalty is not None:
            assert self.top_k is not None, "Error: You have to specify top_k for contrastive search"
            assert self.n_samples == 1, "Error: You can sample new ONLY when restoring, paraphrasing or summarizing"
            assert self.num_beams == 1, "Error: You can use beam search ONLY when restoring, paraphrasing or summarizing"
            assert self.num_beam_groups == 1, "Error: You can use beam search ONLY when restoring, paraphrasing or summarizing"
            assert self.diversity_penalty is None, "Error: You can use diversity penalty ONLY when restoring, paraphrasing or summarizing"
            assert self.temperature is None,  "Error:  You can specify temperature ONLY when restoring, paraphrasing or summarizing"
            assert not self.do_sample, "Error: You can use multinomial sampling ONLY when restoring, paraphrasing or summarizing"
        elif self.top_k is not None:
            assert self.do_sample, "Error: You can specify top_k type ONLY when sampling"
        
        if self.temperature is not None:
            assert self.num_beams>1, "Error: You can specify temperature type ONLY when beams>1"
            assert self.do_sample, "Error: You can specify temperature type ONLY when sampling"

        if self.diversity_penalty is not None:
            #assert self.n_samples>1
            assert self.num_beam_groups > 1, "Error: You have to choose>1 groups for diverse beam decoding"
            assert self.num_beams == self.num_beam_groups, "NOT NESSEASARY"
            assert not self.do_sample, "Error: You cant sample when using diversity"
            assert self.temperature is None, "Error: You cant sample when using diversity"
            assert self.top_k is None, "Error: You cant sample when using diversity"
            assert self.alpha_penalty is None, "Error: You cant use alpha penalty for diverse decoding"
        else:
            assert self.num_beam_groups == 1, "Error: You have to choose>1 groups ONLY for diverse beam decoding"
            assert self.n_samples==1, "Error: You will not get diverse output without diversity decoding"
        
        if self.n_samples>1:
            if self.diversity_penalty is None:
                assert self.do_sample, "Error: You have to sample for diverse output"
        
        if self.augmentation_batch>1:
            assert self.length_multiplier is None, "Error: You can't regulate length multiplier when augmentation batch>1"

        if not self.aug_rest:
            assert self.aug_warming_up == 0, "Error: Warming up for augs is not available in this regime"
            assert not self.renew_aug, "Error: You should turn on instance wise augmentation to renew augs"
            assert self.original_text_p == 1, "Error: You can control the share of original text only in aug_rest"
        else:
            if isinstance(self.original_text_p, float):
                assert self.original_text_p<1, "Error: Augmentations are not used if original text probability equals 1"
            else:
                assert self.original_text_p == 'equal'
            if self.original_text_p==0:
                assert self.aug_warming_up>0
        if self.preprocessing_translator is not None:
            assert self.preprocessing_lang is not None, "Error: Choose lang for preprocessing translator"
        else:
            assert self.preprocessing_translator_batch==1, "Error: You have to specify translator batch only for preprocessing translation"
            assert self.preprocessing_lang is None, "Error: You have to chhose preprocessing lang only in case of preprocessing translation"
            assert self.postprocessing_translator is None

        if aug_controller is not None:
            assert self.aug_controller_regime != '', 'Error: Specify regime for controller'
            if self.aug_controller_regime in ['Max', 'Min']:
                assert self.aug_controller_max > 0, "Error: Specify threshold for max regime"
            else:
                assert self.aug_controller_max == 0, "Error: No need to specify max for this regime"
            if self.aug_controller_regime == 'Diff':
                assert self.controller_max_diff > 0, "Error: Specify threshold for max regime"
            else:
                assert self.controller_max_diff == 0, "Error: No need to specify max for this regime"
        else:
            assert self.aug_controller_regime == '', 'Error: Specigy regime for controller'
            assert self.aug_controller_max == 0, "Error: No need to specify max w/o control"

        if self.triplet:
            assert self.triplet_weight!=0, "Error: You need to specify weight for triplet loss"
            assert self.triplet_type is not None, "Error: You need to specify type for triplet loss"
            if self.triplet_type == 'simcse':
                assert self.simcse_temp is not None, "Error: You need to specify temperature for simcse triplet"
            else:
                assert self.simcse_temp is None, "Error: You need to specify temperature ONLY for simcse triplet"
        else:
            assert self.triplet_weight==0, 'Error: The triplet loss is not added, but triplet weight != 0'
            assert self.triplet_type is  None, 'Error: The triplet loss is not added, but triplet strategy is specified'

        if self.compute_diversity: 
            assert self.diversity_coef!=0, "Error: You need to specify weight for diversity loss"
            assert len(self.diversity_layers)!=0, "Error: You need to specify layers for diversity loss"
            if len(self.diversity_layers)==1:
                assert self.diversity_strategy is None, "Error: You do not need to specify diversity strategy for single layer"
            else:
                assert self.diversity_strategy is not None, "Error: You do not need to specify diversity strategy for multiple layers"
        else:
            assert self.diversity_coef==0, 'Error: Diversity loss is not added, but diversity weight != 0'  
            assert len(self.diversity_layers)==0, 'Error: Diversity loss is not added, but diversity layers are specified'
            assert self.diversity_strategy is None, 'Error: Diversity loss is not added, but diversity strategy is specified'

        if self.pooling == 'prompt-based':
            assert self.prompt is not None, 'Error: You need to specify prompt for prompt-based pooling'
        else:
            assert self.prompt is None, 'Error: You have to specify prompt ONLY for prompt-based pooling'

        if self.pooling in ['prefix', 'prefix+mean']:
            assert self.prefix is not None, 'Error: You need to specify prefix'   
        else: 
            assert self.prefix is None, 'Error: You need to specify prefix ONLY for prefix and prefix+mean pooling'

        if self.pooling in ['soft-prompt', 'soft-prompt+mean', 'soft MLM']:
            assert self.add_tokens > 0, "Error: You have to choose >0 tokens for soft prompt"
        else:
            assert self.add_tokens==0, "Error: You have to choose >0 tokens ONLY for soft prompt"

        if self.pooling in ['MLM', 'soft MLM']:
            assert not self.triplet, 'Error: triplet loss for MLM is not supported'
            if self.pooling == 'MLM':
                assert self.question is not None, "Error: You have to specify question for MLM modelling"
            else:
                assert self.question is None, "Error: You have to specify question ONLY for pure MLM modelling"
        else:
            assert self.question is None,  "Error: You have to specify question ONLY for MLM modelling"

        if self.classifier_layers == []:
            assert self.dropout == 0, "Error: Dropout is not applicable to single layer classificator"
        
        if self.masking>0:
            assert self.masking_type is not None, "Error: You need to specify masking type for masking"
        else:
            assert self.n_safe==0
            assert self.masking_type is None, "Error: You need to specify masking type when NOT masking"
            assert self.masking_warm_up==0, "Error: masking warm up is applicable only when masking>0"
        
        if self.scheduler is not None:
            if self.scheduler == 'default':
                assert isinstance(self.scheduler_fraction, int)
                assert self.scheduler_fraction > 0, "Error: Using default scheduler with 0 fraction"
        else:
            assert self.scheduler_fraction == 0, 'Error: You cant set scheduler fraction without scheduler'
        
        if self.use_pooler:
            assert self.pooling in ['CLS', '[CLS]+mean'], 'Error: You cant use pooler output without CLS pooling'
        
        if self.keep is not None:
            assert self.masking_warm_up>0 or self.aug_warming_up>0, "Error: You need to warup before using instance-wise masking"
            #assert self.masking>0, 'Error: If you wnat to mask certain instances you need to turn on masking'
        else:
            assert not self.renew_idx, "Error: You have to renew idx only for instance wise masking"

        if self.train_in_foreign:
            self.augment = True
            assert self.aug_classes is None
            assert self.preprocessing_translator is not None, "Error: You have to translate to train in foreign"
            assert self.preprocessing_lang is not None, "Error: You have tochoose language for translation"
            assert self.postprocessing_translator is None, "Error: No need to translate back when training in foreign"

        if self.weighting_type is not None:
            assert self.masking > 0, "Error: You can choose weighting type ONLY when masking"
        else:
            assert not self.reverse_weighting, "Error: You cant use reverse weighting without weighting"

        if not self.nli:
            assert self.nli_pos_neg is None, "Error: You have to choose nli pos neg ONLY for NLI"
        if self.n_epochs == 0:
            assert self.nli, "Error: zero-shot classification is availbale only for nli models"
        if not self.etno_replacer:
            assert self.etno_replacer_regime is None, "Error: You cant specify etno replacer regime when not replacing"
        else:
            assert self.etno_replacer_regime is not None, "Error: Please, choose regime of etno replacer" 

        self.device = device

