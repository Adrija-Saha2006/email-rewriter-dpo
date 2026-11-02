"""Step 2 - DPO on (prompt, chosen, rejected) pairs, with LoRA.

Starts from the base model + SFT adapter merged in, then trains a NEW LoRA adapter.
With peft, TRL uses the model with adapter disabled as the frozen reference policy,
so no second copy of the model is needed (saves memory).

NOTE: this is DPO only. There is no reward model and no PPO/RL loop, so this is
NOT full RLHF; it is a direct preference-optimization alternative to it.
Usage: python train_dpo.py
"""
import json, torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
SFT = "out/sft_lora"
OUT = "out/dpo_lora"
BETA = 0.1  # strength of the KL-like pull toward the reference model

tok = AutoTokenizer.from_pretrained(BASE)
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dtype)
model = PeftModel.from_pretrained(model, SFT).merge_and_unload()  # SFT model = reference

rows = [json.loads(l) for l in open("data/train.jsonl")]
ds = Dataset.from_list([{
    "prompt": [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["prompt"]}],
    "chosen": [{"role": "assistant", "content": r["chosen"]}],
    "rejected": [{"role": "assistant", "content": r["rejected"]}]} for r in rows])

cfg = DPOConfig(output_dir=OUT, beta=BETA, num_train_epochs=2, per_device_train_batch_size=4,
                gradient_accumulation_steps=4, learning_rate=5e-5, logging_steps=5,
                save_strategy="no", fp16=torch.cuda.is_available(), report_to="none",
                max_length=512)
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)
trainer.train()
trainer.save_model(OUT)
print("saved", OUT, "- watch rewards/accuracies and rewards/margins in the logs")
