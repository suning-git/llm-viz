#!/usr/bin/env python3
"""
train_2head.py -- retrain gpt-nano on minGPT's sort task with n_head=2 (head_dim=24),
then export the weights to `gpt-nano-sort-model-2head.json` in the *exact same schema*
as bbycroft's original (same keys / shapes / dtype / little-endian-float32 base64).

Why 2 heads?  In the 3D tour, 3 heads x 3 QKV matrices is visually busy; 2 heads is
easier to disambiguate.  Note: changing n_head does NOT change any weight shape --
c_attn stays [144,48], c_proj [48,48] -- it only changes how the 48 q/k/v dims are
split (48/2 = 24 per head).  So the exported JSON is schema-identical to the original.

Faithful to minGPT EXCEPT one deliberate course change: we do PURE next-token training
on EVERY position (no y[:length-1]=-1 masking).  The input half participates in the loss
too, so the model learns "predict-the-next-token at every position; one sequence = T
training examples" -- the paradigm the 3D visualization is meant to teach.  Consequence:
positions 0..4 (predicting the next *random input* digit) learn the ~uniform marginal
(high entropy), while positions >=5 (predicting the sorted output) are high-confidence.

Otherwise faithful to minGPT: SortDataset sampling, N(0,0.02) init with c_proj scaled by
0.02/sqrt(2L), AdamW with decay/no-decay param groups, cross_entropy, grad clip 1.0.

Run (from the minimal_gpt/ dir, venv active):
    python train_2head.py                 # ~a few minutes on CPU
"""
import argparse, base64, json, math, pickle, time
import numpy as np, torch, torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from minimal_gpt import GPT, VOCAB, encode, decode, generate


# ------------------------------------------------------------------ minGPT SortDataset
class SortDataset(Dataset):
    """Random length-6 sequences over {0,1,2}; target = full left-shift (sorted appended).
    COURSE CHANGE vs minGPT: y is NOT masked -- every position is a next-token target.
    Deterministic train/test split via hash of the sequence."""
    def __init__(self, split, length=6, num_digits=3):
        assert split in {"train", "test"}
        self.split, self.length, self.num_digits = split, length, num_digits

    def __len__(self):
        return 10000

    def get_vocab_size(self):
        return self.num_digits

    def get_block_size(self):
        return self.length * 2 - 1

    def __getitem__(self, idx):
        while True:
            inp = torch.randint(self.num_digits, size=(self.length,), dtype=torch.long)
            if torch.rand(1).item() < 0.5:
                if inp.unique().nelement() > self.length // 2:
                    continue                          # bias toward repeated-digit cases
            h = hash(pickle.dumps(inp.tolist()))
            inp_split = "test" if h % 4 == 0 else "train"
            if inp_split == self.split:
                break
        sol = torch.sort(inp)[0]
        cat = torch.cat((inp, sol), dim=0)            # (2*length,)  = [inp(6), sorted(6)]
        x = cat[:-1].clone()                          # (block_size=11,) = inp + sorted[:-1]
        y = cat[1:].clone()                           # PURE next-token target at EVERY pos
        #  (course change) NO  y[:self.length-1] = -1  mask: the input half trains too.
        return x, y


# ------------------------------------------------------------------ minGPT init & optim
def init_weights(model, n_layer):
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    model.apply(_init)
    # scaled init for residual projections (GPT-2 trick)
    for pn, p in model.named_parameters():
        if pn.endswith("c_proj.weight"):
            nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))


def configure_optimizers(model, weight_decay, lr, betas):
    decay, no_decay = set(), set()
    whitelist, blacklist = (nn.Linear,), (nn.LayerNorm, nn.Embedding)
    for mn, m in model.named_modules():
        for pn, _ in m.named_parameters(recurse=False):
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist):
                no_decay.add(fpn)
    pd = {pn: p for pn, p in model.named_parameters()}
    assert not (decay & no_decay) and (decay | no_decay) == set(pd)
    groups = [
        {"params": [pd[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
        {"params": [pd[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


# ------------------------------------------------------------------ batching & eval
def get_batch(ds, bs):
    xs, ys = zip(*[ds[i] for i in range(bs)])         # idx ignored; each item is random
    return torch.stack(xs), torch.stack(ys)


@torch.no_grad()
def eval_accuracy(model, ds, n):
    model.eval()
    ok, seen = 0, set()
    tries = 0
    while len(seen) < n and tries < n * 20:
        tries += 1
        x, _ = ds[0]
        inp = x[: ds.length].tolist()
        key = tuple(inp)
        if key in seen:
            continue
        seen.add(key)
        out = generate(model, inp, ds.length)
        ok += (out == sorted(inp))
    return ok / max(len(seen), 1), len(seen)


# ------------------------------------------------------------------ position behaviour
@torch.no_grad()
def position_stats(model, ds, n=1000):
    """Average max-prob, entropy and next-token accuracy at each of the 11 positions,
    over n random full training sequences.  Shows positions 0..4 (predict next RANDOM
    input digit) are ~uniform/high-entropy, positions >=5 (predict SORTED output) sharp."""
    model.eval()
    T = ds.get_block_size()
    maxp, ent, acc = torch.zeros(T), torch.zeros(T), torch.zeros(T)
    for _ in range(n):
        x, y = ds[0]
        probs = F.softmax(model(x.unsqueeze(0))[0], dim=-1)         # (T, vocab)
        maxp += probs.max(-1).values
        ent += -(probs * probs.clamp_min(1e-12).log()).sum(-1)     # nats
        acc += (probs.argmax(-1) == y).float()
    return maxp / n, ent / n, acc / n


@torch.no_grad()
def position_report(model, input_str):
    """Per-position output distribution for one concrete sequence (all 11 positions)."""
    model.eval()
    inp = encode(input_str)
    cat = inp + sorted(inp)                                          # 12 tokens
    x, y = cat[:-1], cat[1:]                                         # 11 each
    probs = F.softmax(model(torch.tensor([x]))[0], dim=-1)          # (11, 3)
    rows = []
    for i in range(len(x)):
        pr = probs[i]
        ent = float(-(pr * pr.clamp_min(1e-12).log()).sum())
        rows.append(dict(pos=i, in_tok=VOCAB[x[i]], target=VOCAB[y[i]],
                         pred=VOCAB[int(pr.argmax())],
                         pA=float(pr[0]), pB=float(pr[1]), pC=float(pr[2]),
                         entropy=ent, correct=int(pr.argmax()) == y[i]))
    return rows


# ------------------------------------------------------------------ JSON export
def export_json(model, cfg, template_path, out_path):
    """Write the same key list / order / shapes / dtype as `template_path`, but with
    this model's weights (little-endian float32, row-major base64) and cfg.n_head."""
    template = json.load(open(template_path))
    sd = model.state_dict()
    block = cfg["block_size"]
    out = {}
    for k, v in template.items():
        if k == "config":
            out[k] = {**v, "n_head": cfg["n_head"]}    # only n_head changes
            continue
        if k.endswith(".attn.bias"):                   # causal-mask buffer (not a param)
            t = torch.tril(torch.ones(block, block)).view(1, 1, block, block)
        else:
            t = sd[k]
        arr = np.ascontiguousarray(t.detach().numpy()).astype("<f4")   # LE float32, C-order
        assert list(arr.shape) == v["shape"], f"shape mismatch {k}: {arr.shape} vs {v['shape']}"
        out[k] = {"shape": list(arr.shape), "dtype": "torch.float32",
                  "data": base64.b64encode(arr.tobytes()).decode("ascii")}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=4)
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="gpt-nano-sort-model.json")
    ap.add_argument("--out", default="gpt-nano-sort-model-2head.json")
    ap.add_argument("--max_iters", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    train_ds, test_ds = SortDataset("train"), SortDataset("test")

    cfg = dict(model_type="gpt-nano", n_layer=3, n_head=2, n_embd=48,
               vocab_size=train_ds.get_vocab_size(), block_size=train_ds.get_block_size(),
               embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1)
    model = GPT(cfg)
    init_weights(model, cfg["n_layer"])
    print(f"model: n_head={cfg['n_head']} head_dim={cfg['n_embd']//cfg['n_head']} "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    opt = configure_optimizers(model, weight_decay=0.1, lr=args.lr, betas=(0.9, 0.95))

    t0 = time.time()
    model.train()
    for it in range(1, args.max_iters + 1):
        x, y = get_batch(train_ds, args.batch_size)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))  # ALL positions
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % 250 == 0 or it == 1:
            acc, seen = eval_accuracy(model, test_ds, 200)
            print(f"iter {it:4d} | loss {loss.item():.4f} | test sort acc {acc*100:5.1f}% "
                  f"(n={seen}) | {time.time()-t0:5.1f}s")
            model.train()
            if acc >= 0.999 and it >= 1000:
                print("  reached >=99.9% test accuracy, stopping early.")
                break

    # final, larger eval (acceptance #1: greedy 6-step sort accuracy)
    acc, seen = eval_accuracy(model, test_ds, 1000)
    print(f"\nFINAL greedy sort accuracy: {acc*100:.2f}%  over {seen} unique test sequences")

    # acceptance #2: position-behaviour (teaching material)
    maxp, ent, pacc = position_stats(model, train_ds, 2000)
    print("\nper-position behaviour (avg over 2000 random sequences):")
    print("  pos | predicts        | avg max-prob | avg entropy(nats) | next-tok acc")
    for i in range(len(maxp)):
        what = "next INPUT digit" if i < 5 else "sorted OUTPUT   "
        print(f"  {i:3d} | {what} | {maxp[i]:11.3f}  | {ent[i]:16.3f}  | {pacc[i]*100:6.1f}%")
    print(f"  uniform baseline: max-prob=0.333, entropy=ln(3)={math.log(3):.3f} nats")

    print("\nconcrete example 'CBABBC' -> full 11-position output distribution:")
    print("  pos in tgt pred    P(A)   P(B)   P(C)   entropy  correct")
    for r in position_report(model, "CBABBC"):
        print(f"  {r['pos']:3d}  {r['in_tok']}   {r['target']}   {r['pred']}   "
              f"{r['pA']:5.3f}  {r['pB']:5.3f}  {r['pC']:5.3f}   {r['entropy']:6.3f}   "
              f"{'yes' if r['correct'] else 'no'}")

    export_json(model, cfg, args.template, args.out)
    print(f"exported -> {args.out}")
    # sanity: reload with the exact same GPT loader path used by minimal_gpt.py
    from minimal_gpt import load_weights
    m2 = GPT(cfg).eval()
    load_weights(m2, args.out)
    demo = decode(generate(m2, encode("CBABBC"), 6))
    print(f"reload check: 'CBABBC' -> '{demo}' (want 'ABBBCC') "
          f"{'OK' if demo == 'ABBBCC' else 'MISMATCH'}")


if __name__ == "__main__":
    main()
