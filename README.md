# Professional Email Rewriter with DPO

Turn blunt workplace messages into polite, concise, professional ones by teaching a small model *which style people prefer* using Direct Preference Optimization (DPO).

> **Honesty note:** this project uses **DPO** (a preference-optimization method). It has **no reward model and no PPO/RL loop**, so it is *not* full RLHF. DPO is a simpler alternative that optimizes the same preference objective directly.

## The 1-hour plan (free Colab T4 GPU)

| Min | Step | Command |
|---|---|---|
| 0-5 | Setup | `pip install -q "trl>=0.12" peft datasets transformers accelerate` |
| 5-8 | Generate 640 train + 48 held-out pairs | `python gen_data.py` |
| 8-15 | SFT baseline (LoRA, ~3-5 min) | `python train_sft.py` |
| 15-30 | DPO (LoRA, 2 epochs, ~10-15 min) | `python train_dpo.py` |
| 30-38 | Generate outputs for Base / SFT / DPO | `python evaluate.py generate` |
| 38-42 | Auto metrics + preference accuracy | `python evaluate.py metrics` |
| 42-52 | Human rating of 48 blinded pairs (you + a friend) | `python evaluate.py sheet`, fill CSV, `python evaluate.py winrate` |
| 52-60 | Demo on 5 unseen prompts, record video | `python evaluate.py demo` |

In Colab: Runtime > Change runtime type > T4 GPU, upload this folder, then run each command with `!` in front.
No GPU? Use the same scripts with a smaller model (e.g. `HuggingFaceTB/SmolLM2-135M-Instruct`) - change `MODEL`/`BASE` in the 3 scripts.

## Files
- `gen_data.py` - builds the preference dataset (`data/train.jsonl`, `data/heldout.jsonl`, `data/demo_prompts.json`)
- `train_sft.py` - SFT baseline (chosen responses only)
- `train_dpo.py` - DPO with LoRA on top of the SFT model
- `evaluate.py` - generation, metrics, blinded human-rating sheet, win rate, demo

## Dataset format
Each line of `data/train.jsonl`:
```json
{"prompt": "Rewrite ... \"your idea won't work. do it my way.\"",
 "chosen": "Hi Priya, thanks for your proposal ... Could we discuss an alternative?",
 "rejected": "Hi Priya, interesting attempt ... we'll probably go another way.",
 "rejection_type": "passive",
 "rationale": "Chosen states the request directly and respectfully; rejected hides frustration in sarcasm..."}
```
- 640 training pairs over 17 workplace scenarios; same prompt for chosen and rejected.
- Four rejection types (160 each): `curt`, `passive` (passive-aggressive), `verbose` (polite but bloated), `casual` (sloppy/slangy). Each has a rationale.
- Held-out set: 40 new pairs from seen scenarios + 20 pairs from 3 scenarios never seen in training (scope creep, noisy office, credit taken) - this tests generalization.
- **Disclosure:** pairs are template-generated with slot filling (names, documents, days), not written by humans. This gives scale and consistent labels but limited linguistic diversity. Say this in the video.

## Experiments
1. **Base** Qwen2.5-0.5B-Instruct (no fine-tuning).
2. **SFT baseline**: LoRA on chosen responses only.
3. **DPO**: LoRA on chosen vs rejected, starting from SFT.

## Evaluation
- **Auto style metrics** (length, politeness markers, blunt markers, greeting rate) on held-out generations.
- **Preference accuracy**: how often the model gives the chosen answer higher log-probability than the rejected one (implicit-reward check; DPO should raise it over SFT).
- **Human win rate**: blinded A/B sheet (Base vs DPO, random order). Raters mark the better response overall and on politeness, clarity and conciseness. `winrate` reports DPO wins/losses/ties per criterion.
- **Demo**: 5 unseen prompts, baseline vs DPO, with a short explanation of why the preferred one is better.

## Limitations to report
- Politeness and "professional" are subjective and culturally dependent; a small set of raters gives a noisy win rate (n=48, wide confidence interval).
- Template-generated data may teach surface patterns (e.g. always starting with "Hi <name>") rather than real tone understanding.
- The rejected examples are synthetic failure modes, not real human-ranked outputs.
- Keyword style metrics are crude; they can be gamed and miss meaning.
- Risk of over-politeness or dropping facts; a 0.5B model can hallucinate names or details. Check faithfulness manually.
- DPO is sensitive to `beta` and learning rate; we ran one setting, no hyperparameter sweep.
- Not full RLHF: no reward model, no online sampling.

## Video outline (about 6-8 min)
1. **Problem** (30s): blunt messages hurt teamwork; show one example.
2. **Why SFT alone is not enough** (60s): SFT only imitates good examples, it never sees what to avoid; style preferences are relative ("A is better than B"), not a single right answer.
3. **Data format** (60s): prompt / chosen / rejected / rationale; show 2 rows and the 4 rejection types.
4. **RLHF in one picture** (60s): SFT, then train a reward model on human comparisons, then optimize policy with PPO under a KL penalty to the reference model.
5. **DPO intuition** (90s): skip the reward model; loss = -log sigmoid(beta * [(log pi(chosen) - log ref(chosen)) - (log pi(rejected) - log ref(rejected))]). Push up chosen, push down rejected, relative to the reference; beta controls how far we drift.
6. **Pipeline** (60s): LoRA SFT, then DPO with frozen reference (adapter disabled), then evaluation. Mention it is not full RLHF.
7. **Results** (90s): metrics table, human win rate, honest caveats.
8. **Demo** (90s): run `python evaluate.py demo`, explain each of the 5 comparisons.
9. **Limitations** (30s).

## Results (Qwen2.5-0.5B-Instruct, 48 held-out pairs)

| model | avg words | politeness markers | preference accuracy | avg log-prob margin |
|---|---|---|---|---|
| Base | 13.2 | 0.77 | 0.33 | 24.9 |
| SFT | 19.0 | 1.94 | 0.50 | 57.5 |
| DPO | 23.6 | 2.21 | 0.94 | 160.6 |

Preference accuracy = how often the model gives the chosen reply a higher log-probability than the rejected one. DPO raised it from 0.33 (base) and 0.50 (SFT) to 0.94, which is what the DPO objective optimizes. Part of this is the model learning the templates, since held-out pairs share templates with training (only 20 of the 48 come from unseen scenarios).

Human win rate (one rater, blinded A/B, Base vs DPO, n=48): DPO preferred in 47 of 48 pairs overall and on each criterion (politeness, clarity, conciseness). This is a small, subjective sample from a single rater, and DPO outputs are on average longer than base, so the conciseness result should be read with caution.

### Honest findings from the 5-prompt demo
- DPO and SFT learned the style: they open with a greeting and use polite, softer phrasing.
- They also learned template artifacts: they invent a recipient name ("Hi Priya", "Hi Shubham") that is not in the prompt.
- Faithfulness is a problem: some DPO outputs drop or change the original request or add details (for example an invented "Friday" deadline, or a shift-cover refusal turned into rescheduling a meeting). The base model often kept the meaning better but stayed blunt.
- Conclusion: with small template-generated preference data, DPO strongly shifts tone, but without faithfulness-aware pairs it can lose content. Fixes to try: remove names from chosen replies, add pairs where the rejected reply drops the request, and use more diverse real prompts.

## Troubleshooting
- TRL version differences: if `DPOConfig` rejects `max_length`, remove it. If `SFTTrainer` complains about `max_length`, use `max_seq_length`.
- Out of memory: lower `per_device_train_batch_size` to 2 and raise `gradient_accumulation_steps`.
- DPO log to watch: `rewards/accuracies` should rise toward 1.0 and `rewards/margins` should become positive.
