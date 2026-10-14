import torch
import torch.nn.functional as F
import torch.func as tf
from model.model import transformer_forward


def compute_loss(params, idx, targets, n_heads, n_layers, dropout):
    logits = transformer_forward(params, idx, n_heads=n_heads, n_layers=n_layers, dropout=dropout)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)


def _clip(grads, max_norm=1.0):
    total = torch.sqrt(sum((g**2).sum() for g in grads.values() if isinstance(g, torch.Tensor)))
    coef = (max_norm / (total + 1e-6)).clamp(max=1.0)
    return {k: (v * coef if isinstance(v, torch.Tensor) else v) for k, v in grads.items()}


def train_step(params, idx, targets, n_heads, n_layers, dropout,
               state=None, lr=3e-4, b1=0.9, b2=0.95, eps=1e-8, max_norm=1.0):
    if state is None:
        state = {
            "m": {k: torch.zeros_like(v) for k, v in params.items() if isinstance(v, torch.Tensor)},
            "v": {k: torch.zeros_like(v) for k, v in params.items() if isinstance(v, torch.Tensor)},
            "t": 0,
        }
    state["t"] += 1
    t = state["t"]

    grad_fn = tf.grad(compute_loss, argnums=0)
    grads = grad_fn(params, idx, targets, n_heads, n_layers, dropout)
    grads = _clip(grads, max_norm=max_norm)

    with torch.no_grad():
        for k in params.keys():
            if not (isinstance(params[k], torch.Tensor) and params[k].requires_grad):
                continue
            g = grads[k]
            state["m"][k].mul_(b1).add_(g, alpha=1 - b1)
            state["v"][k].mul_(b2).addcmul_(g, g, value=1 - b2)
            mh = state["m"][k] / (1 - b1 ** t)
            vh = state["v"][k] / (1 - b2 ** t)
            params[k] -= lr * mh / (vh.sqrt() + eps)
        loss_val = compute_loss(params, idx, targets, n_heads, n_layers, 0.0)

    return loss_val.item(), state


@torch.no_grad()
def functional_generate(params, idx, max_new_tokens, n_heads, n_layers, temperature=1.0):
    dev = next(iter(params.values())).device
    idx = idx.to(dev)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -512:]
        logits = transformer_forward(params, idx_cond, n_heads=n_heads, n_layers=n_layers, dropout=0.0)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
    return idx