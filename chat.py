import os
import sys
import torch
from tokenizers import Tokenizer

from config.config import (
    TAMPLE, TOK_PATH, WEIGHTS_OUT, CKPT_PREFIX,
    VOCAB_SIZE, HIDDEN_SIZE, N_LAYERS, N_HEADS, MAX_SEQ, FF_MULT,
    TEMPERATURE, MAX_NEW_TOKENS,
)
from model.model import init_weights, device
from model.engine import functional_generate


def find_weights():
    if len(sys.argv) > 1:
        p = sys.argv[1]
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        return p
    if os.path.exists(WEIGHTS_OUT):
        return WEIGHTS_OUT
    ckpts = []
    for name in os.listdir(TAMPLE):
        if name.startswith(CKPT_PREFIX) and name.endswith(".pt"):
            step = int(name.replace(CKPT_PREFIX, "").replace(".pt", ""))
            ckpts.append((step, os.path.join(TAMPLE, name)))
    if not ckpts:
        raise FileNotFoundError(f"Нет весов в {TAMPLE}")
    ckpts.sort()
    return ckpts[-1][1]


weights_path = find_weights()
print(f"Загружаю: {weights_path}")
print(f"vocab={VOCAB_SIZE}, hidden={HIDDEN_SIZE}, layers={N_LAYERS}, temp={TEMPERATURE}")

tok = Tokenizer.from_file(TOK_PATH)
params = init_weights(VOCAB_SIZE, HIDDEN_SIZE, N_LAYERS, MAX_SEQ, FF_MULT)
saved = torch.load(weights_path, map_location=device, weights_only=False)
with torch.no_grad():
    for k in params:
        if k in saved:
            params[k].copy_(saved[k])
        else:
            print(f"  [warn] нет ключа {k}")

print("Готово. Ctrl+C — выход.\n")

while True:
    try:
        prompt = input("> ")
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not prompt.strip():
        continue
    ids = tok.encode(prompt).ids
    if not ids:
        continue
    ctx = torch.tensor([ids], dtype=torch.long, device=device)
    out = functional_generate(
        params, ctx, max_new_tokens=MAX_NEW_TOKENS,
        n_heads=N_HEADS, n_layers=N_LAYERS, temperature=TEMPERATURE,
    )
    full = tok.decode(out[0].tolist())
    print(full[len(prompt):])
    print()