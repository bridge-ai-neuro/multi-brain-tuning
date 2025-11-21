# Configuration file for brain_trainer.py

# -- model args
MODEL_CKPT = {'wav2vec':'facebook/wav2vec2-base-960h',
              'hubert':'facebook/hubert-base-ls960'}
OUT_DIM = 14370 + 15802
NC_THR = 0.4
TRAIN_ROIS = False
RANDOM_TRAIN = False
SUB_NEUR_MASK = None
# -- training args
LORA_RANK = 8
BOTTLENECK_DIM = 200
TRAIN_SUBJECTS = [3, 2, 1]
BASE_LR = 1e-4
LINEAR_LAYER_LR = 1e-4
NUM_EPOCHS = 30
BATCH_SIZE = 128
TR_SIZE = 1.0
DEVICE = 'cuda'

# -- data args
SUBJECT = 3
SAMPLING_RATE = 16000
CHUNKSZ_SEC = 1
CONTEXTSZ_SEC = 0
WAV_PARAMS =  {'sampling_rate': SAMPLING_RATE, 'chunksz_sec': CHUNKSZ_SEC, 'contextsz_sec': CONTEXTSZ_SEC}


# -- logs args
SAVE_DIR = '../outputs/train_logs'
SAVE_NAME = 'mean_mp'
EXP_NAME = 'wav2vec_story_multiple_lora_linear_1e4'

# Evaluation args
CKPT_SAVE_DIR = '../outputs/train_logs'
PREDS_SAVE_DIR = '../outputs/preds_results'
LAYERS_TO_SAVE = [2, 5, 7, 8, 10, 12]
PRED_MODEL_NAME = 'wav2vec'
DESIRED_EPOCHS = [3, 5, 10, 20, 30]  # epochs to evaluate