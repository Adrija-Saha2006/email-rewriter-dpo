"""Evaluation + demo for Base vs SFT vs DPO.

  python evaluate.py generate   # outputs for held-out set + 5 demo prompts -> out/*.json
  python evaluate.py metrics    # auto style metrics + preference accuracy (implicit)
  python evaluate.py sheet      # blinded A/B sheet for HUMAN rating -> out/human_rating.csv
  python evaluate.py winrate    # win rate from the filled CSV
  python evaluate.py demo       # print the 5 unseen prompts: baseline vs DPO
"""
import json, sys, csv, random, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = ("You are a workplace writing assistant. Rewrite the user's message so it is "
          "professional, polite, clear and concise. Keep the original request and facts.")
PROMPT = 'Rewrite this workplace message professionally, politely and concisely:\n\n"{m}"'
dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if dev == "cuda" else torch.float32
tok = AutoTokenizer.from_pretrained(BASE)

def load(name):
    m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dtype).to(dev)
    if name in ("sft", "dpo"):
        m = PeftModel.from_pretrained(m, "out/sft_lora").merge_and_unload()
    if name == "dpo":
        m = PeftModel.from_pretrained(m, "out/dpo_lora").merge_and_unload()
    return m.eval()

def chat(m, prompt):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    with torch.no_grad():
        out = m.generate(ids, max_new_tokens=120, do_sample=False, repetition_penalty=1.05)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

def seq_logp(m, prompt, resp):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    p = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    r = tok(resp + "<|im_end|>", return_tensors="pt", add_special_tokens=False).input_ids[0]
    ids = torch.cat([p, r]).unsqueeze(0).to(dev)
    with torch.no_grad():
        lp = torch.log_softmax(m(ids).logits[0, :-1].float(), -1)
    tgt = ids[0, 1:]
    tok_lp = lp.gather(1, tgt.unsqueeze(1)).squeeze(1)
    return tok_lp[len(p) - 1:].sum().item()

held = [json.loads(l) for l in open("data/heldout.jsonl")]
demo = json.load(open("data/demo_prompts.json"))
MODELS = ["base", "sft", "dpo"]

BLUNT = ["asap", "now", "stop", "fix it", "your fault", "wrong", "unacceptable", "redo", "lol", "u ", "!!"]
POLITE = ["please", "thank", "could you", "would you", "appreciate", "would be able", "hi ", "happy to"]

def style(t):
    tl = t.lower()
    return dict(words=len(t.split()),
                polite=sum(p in tl for p in POLITE),
                blunt=sum(bool(re.search(r"(?<![a-z])"+re.escape(b.strip())+r"(?![a-z])", tl)) for b in BLUNT),
                greeting=int(bool(re.match(r"\s*(hi|hello|dear)\b", tl))))

cmd = sys.argv[1] if len(sys.argv) > 1 else "generate"

if cmd == "generate":
    res = {n: {} for n in MODELS}
    for n in MODELS:
        m = load(n)
        res[n]["held"] = [chat(m, r["prompt"]) for r in held]
        res[n]["demo"] = [chat(m, PROMPT.format(m=d)) for d in demo]
        del m; torch.cuda.empty_cache()
    json.dump(res, open("out/generations.json", "w"), indent=1)
    print("saved out/generations.json")

elif cmd == "metrics":
    g = json.load(open("out/generations.json"))
    print(f"{'model':6} {'words':>6} {'polite':>7} {'blunt':>6} {'greet%':>7} {'pref_acc':>9} {'avg_margin':>11}")
    for n in MODELS:
        s = [style(t) for t in g[n]["held"]]
        avg = lambda k: sum(x[k] for x in s) / len(s)
        m = load(n)
        margins = [seq_logp(m, r["prompt"], r["chosen"]) - seq_logp(m, r["prompt"], r["rejected"]) for r in held]
        acc = sum(x > 0 for x in margins) / len(margins)
        print(f"{n:6} {avg('words'):6.1f} {avg('polite'):7.2f} {avg('blunt'):6.2f} {100*avg('greeting'):6.0f}% {acc:9.2f} {sum(margins)/len(margins):11.2f}")
        del m; torch.cuda.empty_cache()
    print("\npref_acc = fraction of held-out pairs where the model assigns higher log-prob to chosen than rejected.")

elif cmd == "sheet":
    g = json.load(open("out/generations.json"))
    random.seed(0); rows = []
    for i, r in enumerate(held):
        a, b = g["base"]["held"][i], g["dpo"]["held"][i]
        flip = random.random() < 0.5
        rows.append(dict(id=i, original=r["prompt"].split('"')[1], A=b if flip else a, B=a if flip else b,
                         key="A=dpo" if flip else "A=base", better_AorB="", politeness_AorBorTie="",
                         clarity_AorBorTie="", conciseness_AorBorTie=""))
    with open("out/human_rating.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("Fill the *_AorB columns (A/B/Tie), NOT the 'key' column, then run: python evaluate.py winrate")

elif cmd == "winrate":
    rows = list(csv.DictReader(open("out/human_rating.csv")))
    for col in ["better_AorB", "politeness_AorBorTie", "clarity_AorBorTie", "conciseness_AorBorTie"]:
        w = l = t = 0
        for r in rows:
            v = r[col].strip().upper()
            if v not in ("A", "B", "TIE"): continue
            dpo_is = "A" if r["key"] == "A=dpo" else "B"
            if v == "TIE": t += 1
            elif v == dpo_is: w += 1
            else: l += 1
        n = w + l + t
        if n: print(f"{col:24} DPO wins {w}  loses {l}  ties {t}  win-rate(excl. ties)={w/max(w+l,1):.2f}  n={n}")

elif cmd == "demo":
    g = json.load(open("out/generations.json"))
    for i, d in enumerate(demo):
        print(f"\n=== Prompt {i+1}: {d}\n--- BASELINE: {g['base']['demo'][i]}\n--- SFT:      {g['sft']['demo'][i]}\n--- DPO:      {g['dpo']['demo'][i]}")
