from easydict import EasyDict as edict

env_settings = edict()

# env_settings.DEBUG = "/home/data/DIVA/mimic"
env_settings.DEBUG = "home/max/Desktop/MLMI/data"
# env_settings.ROOT = "/home/data/DIVA/mimic/mimic-cxr-jpg/2.0.0"
env_settings.ROOT = "/home/max/Desktop/MLMI/data/mimic-cxr-jpg"
# env_settings.DATA = "/home/data/DIVA/mimic/mimic-cxr-jpg/2.0.0/files"
env_settings.DATA = "/home/max/Desktop/MLMI/data/mimic-cxr-jpg/files"
# env_settings.EXPERIMENTS = "/home/guests/usr_mlmi/arda/Unsupervised-Structured-Reporting-via-Cycle-Consistency/experiments/"
env_settings.EXPERIMENTS = "/home/max/MLMI/Unsupervised-Structured-Reporting-via-Cycle-Consistency/experiments/"
env_settings.CUDA_VISIBLE_DEVICES = 0
env_settings.MASTER_LIST = {
    # "ones": "/home/guests/usr_mlmi/arda/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/data/master_df_ones.csv",
    "ones": "/home/max/MLMI/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/data/master_df_ones.csv",
    # "zeros": "/home/guests/usr_mlmi/arda/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/data/master_df_zeros.csv",
    "zeros": "/home/max/MLMI/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/data/master_df_zeros.csv"
}
env_settings.OCCURENCE_PROBABILITIES = {
    # "ones": "/home/guests/usr_mlmi/arda/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/data/disease_combination_distribution_ones.json",
    "ones" : "/home/max/MLMI/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/data/disease_combination_distribution_ones.json",
    # "zeros": "/home/guests/usr_mlmi/arda/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/data/disease_combination_distribution_zeros.json"
    "zeros" : "/home/max/MLMI/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/data/disease_combination_distribution_zeros.json"
}
# env_settings.CONFIG = "/home/guests/usr_mlmi/arda/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/config.yaml"
env_settings.CONFIG = "/home/max/MLMI/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/config.yaml"
env_settings.PRETRAINED_PATH = {
    # 'ARK': "/home/guests/usr_mlmi/arda/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/pretrained_models",
    'ARK': "/home/max/Desktop/MLMI/Ark/"
}
env_settings.TENSORBOARD_PORT = 1881