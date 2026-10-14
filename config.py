import json
import os

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_BASE, "config", "config.json")

with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    CFG = json.load(_f)

TAMPLE = os.path.join(_BASE, CFG["paths"]["tample"])
os.makedirs(TAMPLE, exist_ok=True)
INPUT_TXT = os.path.join(TAMPLE, CFG["paths"]["input_text"])
TOK_PATH = os.path.join(TAMPLE, CFG["paths"]["tokenizer"])
DATA_BIN = os.path.join(TAMPLE, CFG["paths"]["dataset_bin"])
WEIGHTS_OUT = os.path.join(TAMPLE, CFG["paths"]["weights"])
CKPT_PREFIX = CFG["paths"]["checkpoint_prefix"]

VOCAB_SIZE = CFG["tokenizer"]["vocab_size"]
TOK_CHUNK_MB = CFG["tokenizer"]["chunk_size_mb"]
SPECIAL_TOKENS = CFG["tokenizer"]["special_tokens"]

HIDDEN_SIZE = CFG["model"]["hidden_size"]
N_LAYERS = CFG["model"]["n_layers"]
N_HEADS = CFG["model"]["n_heads"]
MAX_SEQ = CFG["model"]["max_seq"]
DROPOUT = CFG["model"]["dropout"]
FF_MULT = CFG["model"]["ff_mult"]

BATCH_SIZE = CFG["training"]["batch_size"]
SEQ_LEN = CFG["training"]["seq_len"]
LR = CFG["training"]["learning_rate"]
STEPS = CFG["training"]["steps"]
LOG_EVERY = CFG["training"]["log_every"]
CKPT_EVERY = CFG["training"]["checkpoint_every"]
GRAD_CLIP = CFG["training"]["grad_clip"]
ADAM_B1 = CFG["training"]["adam_beta1"]
ADAM_B2 = CFG["training"]["adam_beta2"]
ADAM_EPS = CFG["training"]["adam_eps"]
SEED = CFG["training"]["seed"]

TEMPERATURE = CFG["generation"]["temperature"]
MAX_NEW_TOKENS = CFG["generation"]["max_new_tokens"]