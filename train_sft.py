"""Step 1 - SFT baseline: LoRA fine-tune on the *chosen* responses only.

This is the "SFT-style baseline". It only ever sees good answers, so it never
learns what to avoid. DPO (train_dpo.py) starts from this model.
Usage: python train_sft.py
"""
import json, torch
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = "out/sft_lora"

rows = [json.loads(l) for l in open("data/train.jsonl")]
ds = Dataset.from_list([{"messages": [
    {"role": "system", "content": r["system"]},
    {"role": "user", "content": r["prompt"]},
    {"role": "assistant", "content": r["chosen"]}]} for r in rows])

cfg = SFTConfig(output_dir=OUT, num_train_epochs=1, per_device_train_batch_size=8,
                gradient_accumulation_steps=2, learning_rate=2e-4, logging_steps=10,
                save_strategy="no", fp16=torch.cuda.is_available(), report_to="none",
                max_length=384)
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
trainer = SFTTrainer(model=MODEL, args=cfg, train_dataset=ds, peft_config=lora)
trainer.train()
trainer.save_model(OUT)
print("saved", OUT)
