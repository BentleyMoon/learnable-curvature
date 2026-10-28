# -*- coding: utf-8 -*-
"""Retrofitting learnable-curvature attention into a pretrained model, healed by distillation
(paper/preregistrations/PREREG_RETROFIT_20261021.md; RETRO3 and RETRO4 in the same folder).

The scoring rule is this repository's (src/learnable_curvature_transformer.py, CurvatureDistance): per head, kappa blends
the dot product (kappa = 0), a length-boosted dot product (kappa < 0) and direction only (cosine, kappa > 0).
Retrofit at kappa = 0 changes nothing, so the test switches heads to cosine (kappa = +1, cosine weight 0.995) and
asks whether distillation from the original model heals them, and whether the healed network uses the new rule or
routes around it.

Conditions (Qwen2.5-1.5B, LoRA r16 on q/k/v/o plus a learnable temperature on swapped heads, KL distillation from
the unmodified model on WikiText-2, 1M tokens, seeds 0-1):
  cos_L1     every head of layer 1 switched to cosine (the preprint's single-layer recipe, scaled up)
  cos_Q      every head of every 4th layer (1, 5, ..., 25: 7 of 28 layers) switched to cosine
  uni_Q      the same 7 layers switched to UNIFORM attention (all scores 0): equal disruption, no geometry
  learn_all  every head gets a learnable kappa from 0 (does the network choose a geometry when free to?)
Evaluation on held-out WikiText-2 (last 10% of the text): perplexity of the teacher, of the swapped model before
healing, after healing, and two ablations on the healed model -- (a) the swapped heads reverted to the dot product
with healed weights, (b) the swapped heads' outputs zeroed -- with (b) also applied to the teacher.

  python exp_retrofit.py --smoke          checks the hook reproduces the original model at kappa = 0
  R_COND=cos_Q R_SEEDS=0,1 python exp_retrofit.py
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("R_MODEL", "Qwen/Qwen2.5-1.5B")
DATA = os.environ.get("R_DATA", os.path.join(HERE, "..", "..", "data", "wikitext2_all.txt"))   # scripts/get_data.py
OUT = os.path.join(HERE, os.environ.get("R_OUT", "results"))   # RETRO2: SmolLM writes results_smol
SEQ, BATCH, MICRO = 256, 8, 2
# RETRO4: R_SEQ sets the healing sequence length only; tokens per step (2,048) and per micro-batch (512) are kept,
# so the step count is unchanged. Perplexity and retrieval evaluation stay at 256-token windows.
TSEQ = int(os.environ.get("R_SEQ", str(SEQ)))
TBATCH, TMICRO = BATCH * SEQ // TSEQ, max(1, MICRO * SEQ // TSEQ)
BUDGET = int(os.environ.get("R_BUDGET", "1000000"))
EVAL_SEQS = int(os.environ.get("R_EVAL", "200"))   # 200 x 256 = 51,200 held-out tokens
STATE = {"off": False}                # teacher pass: every head on the plain dot product, LoRA disabled


def curv_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    import torch
    import torch.nn.functional as F
    g = getattr(module, "num_key_value_groups", 1)
    if g > 1:
        key = key.repeat_interleave(g, dim=1)
        value = value.repeat_interleave(g, dim=1)
    dot = torch.matmul(query, key.transpose(-2, -1)) * scaling
    kappa = getattr(module, "curv_kappa", None)
    mode = getattr(module, "curv_mode", None)
    if STATE["off"] or mode is None:
        scores = dot
    elif mode == "uniform":
        scores = torch.zeros_like(dot)
    else:
        d = query.size(-1)
        k3 = (kappa.float() * 3.0).view(1, -1, 1, 1)
        hyp_w, sph_w = torch.tanh(-k3).clamp(min=0), torch.tanh(k3).clamp(min=0)
        euc_w = 1.0 - hyp_w - sph_w
        qn, kn = query.float().norm(dim=-1, keepdim=True), key.float().norm(dim=-1, keepdim=True)
        hyp = dot.float() * (1.0 + 0.5 * qn * kn.transpose(-2, -1) / math.sqrt(d))
        cos = torch.matmul(F.normalize(query.float(), dim=-1), F.normalize(key.float(), dim=-1).transpose(-2, -1))
        scores = euc_w * dot.float() + hyp_w * hyp + sph_w * cos * math.sqrt(d)
        tau = getattr(module, "curv_logtau", None)
        if tau is not None:
            scores = scores / F.softplus(tau.float()).view(1, -1, 1, 1)
    S = scores.size(-1)
    causal = torch.ones(S, S, dtype=torch.bool, device=scores.device).tril()
    scores = scores.float().masked_fill(~causal, float("-inf"))
    if attention_mask is not None and attention_mask.dtype != torch.bool:
        scores = scores + attention_mask[..., -S:].float()
    attn = torch.softmax(scores, dim=-1).to(value.dtype)
    out = torch.matmul(attn, value)
    if getattr(module, "curv_zero", False):
        out = out * 0
    return out.transpose(1, 2).contiguous(), attn


def install(model, cond):
    import torch
    from transformers import AttentionInterface
    AttentionInterface.register("curv", curv_attention)
    model.config._attn_implementation = "curv"
    layers = model.model.layers
    H = model.config.num_attention_heads
    swapped = {"cos_L1": [1], "cos_Q": list(range(1, len(layers), 4)), "uni_Q": list(range(1, len(layers), 4)),
               "learn_all": list(range(len(layers))), "none": []}[cond]
    dev = next(model.parameters()).device
    for i in swapped:
        att = layers[i].self_attn
        if cond == "uni_Q":
            att.curv_mode = "uniform"
        elif cond == "learn_all":
            att.curv_mode = "curv"
            # Not exactly 0: the blend's clamp has a kink at 0 where MPS returns a zero gradient (checked 10-21).
            g = torch.Generator().manual_seed(1000 + i)
            att.curv_kappa = torch.nn.Parameter((torch.randn(H, generator=g) * 1e-3).to(dev))
        else:
            att.curv_mode = "curv"
            att.curv_kappa = torch.nn.Parameter(torch.ones(H, device=dev), requires_grad=False)
            att.curv_logtau = torch.nn.Parameter(torch.full((H,), math.log(math.e - 1), device=dev))  # softplus = 1
    return swapped


def batches(ids, seed, n_tokens):
    import random
    rng = random.Random("retrofit-%d" % seed)
    used = 0
    while used < n_tokens:
        starts = [rng.randrange(0, len(ids) - TSEQ - 1) for _ in range(TBATCH)]
        yield [ids[s:s + TSEQ] for s in starts]
        used += TBATCH * TSEQ


def get_layers(model):
    m = model
    for attr in ("base_model", "model", "model"):
        if hasattr(m, "layers"):
            break
        m = getattr(m, attr)
    return m.layers if hasattr(m, "layers") else m.model.layers


def ppl(model, ev, dev, zero=None, revert=None, uniform=None):
    """Held-out perplexity; `zero` = layer indices whose swapped heads are zeroed; `revert` = layers reverted to dot."""
    import torch
    layers = get_layers(model)
    saved = {}
    for i in (zero or []):
        layers[i].self_attn.curv_zero = True
    for i in (revert or []):
        saved[i] = getattr(layers[i].self_attn, "curv_mode", None)
        layers[i].self_attn.curv_mode = None
    for i in (uniform or []):
        saved[i] = getattr(layers[i].self_attn, "curv_mode", None)
        layers[i].self_attn.curv_mode = "uniform"
    tot, n = 0.0, 0
    with torch.no_grad():
        for j in range(0, len(ev), 8):
            x = torch.tensor(ev[j:j + 8], device=dev)
            lg = model(input_ids=x).logits.float()
            tot += float(torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1), reduction="sum"))
            n += x[:, 1:].numel()
    for i in (zero or []):
        layers[i].self_attn.curv_zero = False
    for i, m in saved.items():
        layers[i].self_attn.curv_mode = m
    return round(math.exp(tot / n), 4)


# RETRO3 (PREREG_RETRO3_20261027.md): retrieval tests, the capability most tied to sharp attention. Generated here
# from a fixed word list and seed; no dataset. Off unless R_RETRIEVAL=1, and run only after the perplexity evals.
RETRIEVAL = os.environ.get("R_RETRIEVAL") == "1"
WORDS = list(dict.fromkeys((
    "apple river stone cloud maple tiger candle mirror garden pencil rocket violin ladder carpet orange bottle "
    "hammer island jacket kettle lemon magnet needle ocean pepper quilt rabbit saddle tunnel umbrella valley window "
    "zebra anchor basket cactus dragon engine falcon guitar harbor jungle kitten lantern meadow napkin oyster parrot "
    "ribbon salmon teapot velvet walnut button cherry desert feather glacier helmet insect kernel lizard marble "
    "nickel olive pillow pirate raven saucer shovel spider thunder tulip wagon wizard badger beacon blanket bridge "
    "camel castle cobra copper cotton crystal dolphin donkey eagle forest ginger goblet hollow honey jasmine knight "
    "lobster mango mitten monkey nectar onion orchid panda pearl planet pumpkin puzzle rhino rubber sailor shadow "
    "silver spoon statue sugar summit timber tomato tower trumpet turtle walrus whistle willow winter yogurt").split()))


def retrieval_items():
    """Key-value recall at 8, 32 and 64 pairs (measured 10-28: about 84, 300 and 587 SmolLM tokens) and copying a random
    word list of 40 or 150 words (induction), 100 items each."""
    import random
    rng = random.Random("retro3-retrieval-20261027")
    items = []
    for n in (8, 32, 64):
        for _ in range(100):
            keys, vals = rng.sample(WORDS, n), rng.sample(range(1000, 10000), n)
            j = rng.randrange(n)
            prompt = "Codes:\n" + "".join("%s: %d\n" % kv for kv in zip(keys, vals)) + keys[j] + ":"
            items.append(("kv%d" % n, prompt, " %d" % vals[j]))
    for n in (40, 150):
        for _ in range(100):
            seq = " ".join(rng.choice(WORDS) for _ in range(n))
            items.append(("ind%d" % n, seq + " |", " " + seq))
    if os.environ.get("R_KV100") == "1":   # RETRO4: about 911 tokens, past a 512-token healing length; drawn after
        for _ in range(100):                # the RETRO3 items, so those stay identical
            keys, vals = rng.sample(WORDS, 100), rng.sample(range(1000, 10000), 100)
            j = rng.randrange(100)
            prompt = "Codes:\n" + "".join("%s: %d\n" % kv for kv in zip(keys, vals)) + keys[j] + ":"
            items.append(("kv100", prompt, " %d" % vals[j]))
    return items


def downstream(model, tok, dev, items, ev, revert=None, uniform=None):
    """kv: exact match of the value (every target token is the argmax; the same as greedy decoding reproducing it).
    ind: share of the copy's tokens predicted, after its first. wiki: held-out perplexity on tokens that already
    appeared earlier in their 256-token window (rep) and on the rest (new)."""
    import torch
    import torch.nn.functional as F
    layers = get_layers(model)
    saved = {}
    for i in (revert or []):
        saved[i] = getattr(layers[i].self_attn, "curv_mode", None)
        layers[i].self_attn.curv_mode = None
    for i in (uniform or []):
        saved[i] = getattr(layers[i].self_attn, "curv_mode", None)
        layers[i].self_attn.curv_mode = "uniform"
    hits, cnt = {}, {}
    with torch.no_grad():
        for fam, prompt, target in items:
            p = tok(prompt)["input_ids"]
            t = tok(target, add_special_tokens=False)["input_ids"]
            pred = model(input_ids=torch.tensor([p + t], device=dev)).logits[0, len(p) - 1:-1].argmax(-1).tolist()
            ok = [a == b for a, b in zip(pred, t)]
            hits[fam] = hits.get(fam, 0.0) + (float(all(ok)) if fam.startswith("kv") else sum(ok[1:]) / max(1, len(ok) - 1))
            cnt[fam] = cnt.get(fam, 0) + 1
        tot, n = {"rep": 0.0, "new": 0.0}, {"rep": 0, "new": 0}
        for j in range(0, len(ev), 8):
            x = torch.tensor(ev[j:j + 8], device=dev)
            lg = model(input_ids=x).logits.float()
            ce = F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1), reduction="none").view(x.size(0), -1)
            S = x.size(1) - 1
            earlier = torch.ones(S, S, dtype=torch.bool, device=dev).tril()        # source s <= t precedes target t+1
            rep = ((x[:, 1:, None] == x[:, None, :-1]) & earlier).any(-1)
            for k, m in (("rep", rep), ("new", ~rep)):
                tot[k] += float(ce[m].sum()); n[k] += int(m.sum())
    for i, m in saved.items():
        layers[i].self_attn.curv_mode = m
    out = {f: round(hits[f] / cnt[f], 4) for f in hits}
    out.update({"wiki_%s" % k: round(math.exp(tot[k] / n[k]), 4) for k in tot}, wiki_rep_share=round(n["rep"] / (n["rep"] + n["new"]), 4))
    return out


def run(cond, seed):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(seed)
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    text = open(DATA, encoding="utf-8").read()
    cut = int(len(text) * 0.9)
    tr_ids = tok(text[:cut])["input_ids"]
    ev_all = tok(text[cut:])["input_ids"]
    ev = [ev_all[i:i + SEQ] for i in range(0, len(ev_all) - SEQ, SEQ)][:EVAL_SEQS]
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16).to(dev)
    swapped = install(model, cond)
    res = {"cond": cond, "seed": seed, "model": MODEL, "swapped_layers": swapped, "budget": BUDGET}
    STATE["off"] = True
    res["ppl_teacher"] = ppl(model, ev, dev)
    res["ppl_teacher_zero"] = ppl(model, ev, dev, zero=swapped) if cond != "learn_all" else None
    STATE["off"] = False
    res["ppl_swapped"] = ppl(model, ev, dev)
    if RETRIEVAL:
        items = retrieval_items()
        STATE["off"] = True
        res["ret_teacher"] = downstream(model, tok, dev, items, ev)
        STATE["off"] = False
        if cond != "none":
            res["ret_swapped"] = downstream(model, tok, dev, items, ev)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    # peft freezes every non-LoRA parameter; the temperatures (and kappa in learn_all) must train with the adapter
    for n, p in model.named_parameters():
        if "curv_logtau" in n or ("curv_kappa" in n and cond == "learn_all"):
            p.data = p.data.float()
            p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2e-4)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    t0, steps, last = time.time(), 0, None
    model.train()
    for b in batches(tr_ids, seed, BUDGET):
        x = torch.tensor(b, device=dev)
        for j in range(0, TBATCH, TMICRO):
            xs = x[j:j + TMICRO]
            with torch.no_grad(), model.disable_adapter():
                STATE["off"] = True
                t_logp = torch.log_softmax(model(input_ids=xs).logits.float(), -1)
                STATE["off"] = False
            s_logp = torch.log_softmax(model(input_ids=xs).logits.float(), -1)
            kl = torch.nn.functional.kl_div(s_logp, t_logp, log_target=True, reduction="batchmean") / TSEQ
            (kl * TMICRO / TBATCH).backward()
            last = float(kl.detach())
        opt.step(); opt.zero_grad(); steps += 1
    model.eval()
    model.gradient_checkpointing_disable()
    res.update(train_s=round(time.time() - t0, 1), steps=steps, last_kl=last)
    res["ppl_healed"] = ppl(model, ev, dev)
    if cond in ("cos_L1", "cos_Q"):
        res["ppl_healed_reverted"] = ppl(model, ev, dev, revert=swapped)
        res["ppl_healed_uniform"] = ppl(model, ev, dev, uniform=swapped)     # routed around? cosine -> uniform
    if cond != "learn_all":
        res["ppl_healed_zero"] = ppl(model, ev, dev, zero=swapped)
    if RETRIEVAL:
        res["ret_healed"] = downstream(model, tok, dev, items, ev)
        if cond in ("cos_L1", "cos_Q"):
            res["ret_healed_reverted"] = downstream(model, tok, dev, items, ev, revert=swapped)
            res["ret_healed_uniform"] = downstream(model, tok, dev, items, ev, uniform=swapped)
    res["kappa"] = {n: [round(float(v), 4) for v in p.detach().cpu()] for n, p in model.named_parameters() if "curv_kappa" in n}
    res["tau"] = {n: [round(float(torch.nn.functional.softplus(v)), 4) for v in p.detach().cpu()] for n, p in model.named_parameters() if "curv_logtau" in n}
    return res


def smoke():
    """At kappa = 0 the hook must reproduce the original model; with heads swapped it must not."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    x = tok("The quick brown fox jumps over the lazy dog because it", return_tensors="pt")["input_ids"]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    x = x.to(dev)
    ref = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager").to(dev)
    with torch.no_grad():
        a = ref(input_ids=x).logits.float()
    install(ref, "learn_all")
    STATE["off"] = True
    with torch.no_grad():
        b0 = ref(input_ids=x).logits.float()
    STATE["off"] = False
    with torch.no_grad():
        b = ref(input_ids=x).logits.float()
    for i in range(1, len(get_layers(ref)), 4):
        get_layers(ref)[i].self_attn.curv_kappa.data.fill_(1.0)
    with torch.no_grad():
        c = ref(input_ids=x).logits.float()
    print("eager vs hook(dot): %.3f | hook(dot) vs hook(kappa=0): %.3f | hook(dot) vs 7 cosine layers: %.2f | top-1 agree eager/hook %.2f"
          % ((a - b0).abs().max(), (b0 - b).abs().max(), (b0 - c).abs().max(), float((a.argmax(-1) == b0.argmax(-1)).float().mean())))


def main():
    if "--smoke" in sys.argv:
        return smoke()
    os.makedirs(OUT, exist_ok=True)
    cond = os.environ["R_COND"]
    for seed in [int(s) for s in os.environ.get("R_SEEDS", "0,1").split(",")]:
        p = os.path.join(OUT, "%s_seed%d.json" % (cond, seed))
        if os.path.exists(p):
            continue
        r = run(cond, seed)
        json.dump(r, open(p, "w"), indent=1)
        print(time.strftime("%m-%d %H:%M:%S"), "done", cond, seed, {k: r[k] for k in r if k.startswith("ppl")}, flush=True)


if __name__ == "__main__":
    main()
