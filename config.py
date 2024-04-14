import jax.numpy as jnp

PARAMTER_DTYPE = jnp.bfloat16

DEFAULT_IMAGE_PATCH_TOKEN = f"<im_patch>"
DEFAULT_IM_START_TOKEN = f"<im_start>"
DEFAULT_IM_END_TOKEN = f"<im_end>"
DEFAULT_IM_COL_TOKEN = f"<im_col>"

EXTRA_TOKENS = (DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_COL_TOKEN)

# ****************************************************************
# Do not change !!!
# ****************************************************************
TFDS_DATA_DIR = "gs://unified-io-2-us-east/"
MULTITASK_TFDS_DATA_DIR = f"{TFDS_DATA_DIR}multitask-datasets"
# ****************************************************************
# Do not change !!!
# ****************************************************************

BASE_IMAGE_INPUT_SIZE = [336, 336]
BASE_IMAGE_INPUT_D = 14
MAX_NUM_PATCHES = 4

# Controls data augmentation
RANDOM_SCALE_MAX = 1.1
RANDOM_SCALE_MIN = 0.9
RANDOM_SCALE_RATIO = 0.5


PROPMPT_MANAGER = {}

PROPMPT_MANAGER['null'] = None

PROPMPT_MANAGER['llama2'] = {}

PROPMPT_MANAGER['llama2']['SYS_PROMPT'] =  "You are a helpful, respectful and honest assistant. \
    Always answer as helpfully as possible, while being safe.  \
    Your answers should not include any harmful, unethical, \
    racist, sexist, toxic, dangerous, or illegal content. \
    Please ensure that your responses are socially unbiased \
    and positive in nature. If a question does not make any sense, \
    or is not factually coherent, explain why instead of answering \
    something not correct. If you don't know the answer to a question, \
    please don't share false information."

PROPMPT_MANAGER['llama2']['B_SYS'] = '<<SYS>>\n'
PROPMPT_MANAGER['llama2']['E_SYS'] = '\n<</SYS>>\n\n'

PROPMPT_MANAGER['llama2']['B_INST'] = '[INST]'
PROPMPT_MANAGER['llama2']['E_INST']  = '[/INST]'

PROPMPT_MANAGER['llama2']['SYS_PREFIX'] = PROPMPT_MANAGER['llama2']['B_SYS'] + \
                                        PROPMPT_MANAGER['llama2']['SYS_PROMPT'] + \
                                        PROPMPT_MANAGER['llama2']['E_SYS']
                                        
                                        
PROPMPT_MANAGER['mistral'] = {}

PROPMPT_MANAGER['mistral']['B_INST'] = '[INST]'
PROPMPT_MANAGER['mistral']['E_INST']  = '[/INST]'


PROPMPT_MANAGER['vicuna_v1'] = {}

PROPMPT_MANAGER['vicuna_v1']['SYS_PROMPT'] =  "A chat between a curious user and an \
    artificial intelligence assistant. The assistant gives helpful, detailed, and \
    polite answers to the user's questions."

PROPMPT_MANAGER['vicuna_v1']['B_SYS'] = ''
PROPMPT_MANAGER['vicuna_v1']['E_SYS'] = ''

PROPMPT_MANAGER['vicuna_v1']['B_INST'] = 'USER: '
PROPMPT_MANAGER['vicuna_v1']['E_INST'] = 'ASSISTANT: '

PROPMPT_MANAGER['vicuna_v1']['SEP']  = ' '
PROPMPT_MANAGER['vicuna_v1']['SEP2']  = '</s>'



PROPMPT_MANAGER['vicuna_v1']['SYS_PREFIX'] = PROPMPT_MANAGER['vicuna_v1']['B_SYS'] + \
                                        PROPMPT_MANAGER['vicuna_v1']['SYS_PROMPT'] + \
                                        PROPMPT_MANAGER['vicuna_v1']['E_SYS']
                                        
