from types import SimpleNamespace

cfg = SimpleNamespace(**{})

cfg.exp_name = 'ait_bird_local'
cfg.apex = True  # [True, False]

######################
# Globals #
######################
cfg.seed = 42
cfg.epochs = 200
cfg.external = True  # [True, False]
cfg.use_sampler = False  # [True, False]

######################
# Data #
######################
cfg.train_audio_dir = "./data/sounds/"
cfg.train_df_dir = "./data/dataframes/"
cfg.train_pkl_file = "ait_train_meta.pickle"
cfg.nocall_pkl_file = "ff1010bird_metadata_v1_pseudo.pickle"
cfg.short_noise_dir = "./data/external/esc50/use_label"
cfg.background_noise_dir = "./data/external/zenodo_nocall_30sec"

######################
# Dataset #
######################
cfg.period = 20  # [5, 10, 20, 30]
cfg.frames = -1  # [-1, 480000, 640000, 960000]

cfg.use_pcen = False
cfg.n_mels = 128  # [64, 128, 224, 256]
cfg.fmin = 20  # [20, 50]
cfg.fmax = 16000  # [14000, 16000]
cfg.n_fft = 2048  # [1024, 2048]
cfg.hop_length = 512  # [320, 512]
cfg.sample_rate = 32000
cfg.secondary_coef = 0.0

cfg.target_columns = [
    'Abroscopus-superciliaris', 'Cyornis-whitei', 'Lanius-Schach', 'Psilopogon-asiaticus',
    'Alcedo-atthis', 'Dicrurus-leucophaeus', 'Merops-leschenaulti', 'Psilopogon-franklinii',
    'Alophoixus-pallidus', 'Dicrurus-remifer', 'Merops-orientalis', 'Psilopogon-haemacephalus',
    'Anthus-hodgsoni', 'Dinopium-javanense', 'Myiomela-leucura', 'Psilopogon-lineatus',
    'Apus-cooki', 'Erpornis-zantholeuca', 'Nyctyornis-athertoni', 'Psilopogon-virens',
    'Bambusicola-fytchii', 'Eudynamys-scolopaceus', 'Parus-minor', 'Pycnonotus-aurigaster',
    'Blythipicus-pyrrhotis', 'Eurystomus-orientalis', 'Pericrocotus-speciosus', 'Saxicola-stejnegeri',
    'Cacomantis-merulinus', 'Gallus-gallus', 'Phaenicophaeus-tristis', 'Sitta-frontalis',
    'Cacomantis-sonneratii', 'Glaucidium-cuculoides', 'Phoenicurus-auroreus', 'Spilopelia-chinensis',
    'Centropus-bengalensis', 'Halcyon-smyrnensis', 'Phyllergates-cucullatus', 'Surniculus-lugubris',
    'Centropus-sinensis', 'Harpactes-erythrocephalus', 'Phylloscopus-claudiae', 'Turnix-suscitator',
    'Ceryle-rudis', 'Harpactes-oreskios', 'Phylloscopus-humei', 'Turnix-tanki',
    'Chrysococcyx-maculatus', 'Hierococcyx-sparverioides', 'Phylloscopus-inornatus', 'Upupa-epops',
    'Chrysocolaptes-guttacristatus', 'Hirundo-rustica', 'Phylloscopus-omeiensis', 'Urosphena-squameiceps',
    'Copsychus-malabaricus', 'Hypothymis-azurea', 'Phylloscopus-ricketti', 'Yungipicus-canicapillus',
    'Coracias-benghalensis', 'Hypsipetes-leucocephalus', 'Phylloscopus-tephrocephalus',
    'Culicicapa-ceylonensis', 'Ixos-mcclellandii', 'Picumnus-innominatus', 'nocall'
]

cfg.bird2id = {b: i for i, b in enumerate(cfg.target_columns)}
cfg.id2bird = {i: b for i, b in enumerate(cfg.target_columns)}

######################
# Loaders #
######################
cfg.loader_params = {
    "train": {
        "batch_size": 32,
        "pin_memory": True,
        "num_workers": 8,
        "drop_last": True,
        "shuffle": True if not cfg.use_sampler else False
    },
    "valid": {
        "batch_size": 64,
        "pin_memory": True,
        "num_workers": 8,
        "shuffle": False
    }
}

######################
# Model #
######################
cfg.backbone = 'eca_nfnet_l0'
cfg.use_imagenet_weights = True
cfg.num_classes = len(cfg.target_columns)
cfg.in_channels = 1
cfg.lr_max = 2.5e-4
cfg.lr_min = 1e-7
cfg.weight_decay = 1e-6
cfg.max_grad_norm = 10
cfg.early_stopping = 20
cfg.mixup_p = 1.0

cfg.pretrained_weights = True
cfg.pretrained_path = './pretrained/fold_0_model.bin'
cfg.model_output_path = f"./out_weights/{cfg.exp_name}_{cfg.backbone}"
