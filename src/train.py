# src/train.py
"""Model construction and fine-tuning utilities."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

__all__ = [
    "ModelBuilder",
    "TrainerWrapper",
]


class ModelBuilder:
    """Load a base model / tokenizer pair and (optionally) add a LoRA head."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir

    # ---------------------------------------------------------------------
    # Model loading
    # ---------------------------------------------------------------------
    def load_base(self, model_id: str, *, quant: bool = False):
        """Load a pretrained causal-LM in float16 or 8-bit."""

        bnb_cfg = None
        if quant:
            # bitsandbytes 8-bit quantisation
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["lm_head"])

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=bnb_cfg,
            cache_dir=self.cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=self.cache_dir)
        # ensure padding token is set
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------
    @staticmethod
    def add_lora(model, *, r: int = 16, alpha: int = 32, dropout: float = 0.05):
        """Wrap the model with PEFT / LoRA adapters."""

        lora_cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=dropout,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()  # log trainable params for debug
        return model


class TrainerWrapper:
    """Light wrapper around 🤗 Trainer to keep main.py readable."""

    def __init__(self, model, tokenizer, *, output_dir: Path, args_cfg: Dict[str, Any]):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.args_cfg = args_cfg

        self.tr_args = TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=args_cfg.get("batch", 1),
            gradient_accumulation_steps=args_cfg.get("grad_accum", 4),
            learning_rate=args_cfg.get("lr", 2e-5),
            num_train_epochs=args_cfg.get("epochs", 1.0),
            fp16=True,
            bf16=False,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            report_to=["none"],
        )

        self.trainer = Trainer(
            model=model,
            args=self.tr_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            train_dataset=None,  # will be filled in train()
            eval_dataset=None,
            tokenizer=tokenizer,
        )

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------
    def train(self, train_ds, eval_ds):
        """Run one training / evaluation loop and return the trained model."""

        self.trainer.train_dataset = train_ds
        self.trainer.eval_dataset = eval_ds
        self.trainer.train()
        return self.trainer.model
