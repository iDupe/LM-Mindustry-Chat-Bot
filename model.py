import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


def init_weights(vocab_size, s_hidden, n_layers, max_seq, ff_mult=2.6):
    s_ff = int(s_hidden * ff_mult)
    res_scale = (2 * n_layers) ** -0.5

    def xavier(shape, scale=1.0):
        t = torch.empty(shape)
        nn.init.xavier_uniform_(t)
        return t * scale

    params = {
        "token_emb": torch.randn(vocab_size, s_hidden) * 0.02,
        "pos_emb": torch.randn(max_seq, s_hidden) * 0.02,
    }
    for i in range(n_layers):
        p = f"L{i}"
        params[f"{p}_ln1_g"] = torch.ones(s_hidden)
        params[f"{p}_w_qkv"] = xavier((s_hidden, s_hidden * 3))
        params[f"{p}_out_proj"] = xavier((s_hidden, s_hidden), scale=res_scale)
        params[f"{p}_ln2_g"] = torch.ones(s_hidden)
        params[f"{p}_swi_w1"] = xavier((s_hidden, s_ff))
        params[f"{p}_swi_w2"] = xavier((s_hidden, s_ff))
        params[f"{p}_swi_w3"] = xavier((s_ff, s_hidden), scale=res_scale)
    params["ln_f_g"] = torch.ones(s_hidden)

    for k in params:
        params[k] = params[k].to(device).requires_grad_(True)
    return params


def rms_norm(x, gamma, eps=1e-5):
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, (gamma.shape[-1],), weight=gamma, eps=eps)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma


def swiglu_func(x, w1, w2, w3):
    return (F.silu(x @ w2) * (x @ w1)) @ w3


def block_func(x, p, prefix, n_heads, dropout=0.0):
    B, T, C = x.shape

    norm_x = rms_norm(x, p[f"{prefix}_ln1_g"])
    qkv = norm_x @ p[f"{prefix}_w_qkv"]
    q, k, v = qkv.view(B, T, 3, n_heads, C // n_heads).unbind(2)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    attn = attn.transpose(1, 2).reshape(B, T, C) @ p[f"{prefix}_out_proj"]
    if dropout > 0:
        attn = F.dropout(attn, p=dropout, training=True)
    x = x + attn

    norm_x = rms_norm(x, p[f"{prefix}_ln2_g"])
    ffn = swiglu_func(norm_x, p[f"{prefix}_swi_w1"], p[f"{prefix}_swi_w2"], p[f"{prefix}_swi_w3"])
    if dropout > 0:
        ffn = F.dropout(ffn, p=dropout, training=True)
    x = x + ffn
    return x


def transformer_forward(params, idx, n_heads, n_layers, dropout=0.0):
    B, T = idx.shape
    x = F.embedding(idx, params["token_emb"]) + params["pos_emb"][:T]
    for i in range(n_layers):
        x = block_func(x, params, f"L{i}", n_heads, dropout=dropout)
    x = rms_norm(x, params["ln_f_g"])
    return x @ params["token_emb"].T