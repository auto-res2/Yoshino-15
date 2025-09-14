# src/train.py
"""Model construction and fine-tuning utilities (iteration-8)."""
from __future__ import annotations

import os
from inspect import signature
from pathlib import Path
from typing import Dict, Any, List, Union

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    DataCollatorForLanguageModeling,
)
# -----------------------------------------------------------------------------
# IMPORTANT
# -----------------------------------------------------------------------------
# We explicitly import ``TrainingArguments`` from the dedicated sub-module to
# minimise the risk of namespace pollution.  However, some environments may ship
# a stripped-down variant that lacks fields such as ``evaluation_strategy`` or
# ``remove_unused_columns``.  We therefore *optimistically* add optional kwargs
# and fall back gracefully if they are rejected, guaranteeing forward-
# compatibility while maintaining our fail-fast policy for unrelated errors.
# -----------------------------------------------------------------------------
from transformers.training_args import TrainingArguments  # noqa: E402

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
            dtype=torch.float16,  # ``torch_dtype`` is deprecated from HF 4.41 → use ``dtype``
            device_map="auto",
            quantization_config=bnb_cfg,
            cache_dir=self.cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=self.cache_dir)
        # ensure padding token is set ------------------------------------------------------
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    # ------------------------------------------------------------------
    # LoRA helpers
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
        self.tokenizer = tokenizer  # keep for on-the-fly tokenisation

        # ------------------------------------------------------------------
        # Build a kwargs dict and *only* pass fields the detected
        # ``TrainingArguments`` __init__ actually supports.  This makes the
        # suite resilient to API differences across transformer versions.
        # ------------------------------------------------------------------
        base_kwargs: Dict[str, Any] = {
            "output_dir": str(self.output_dir),
            "per_device_train_batch_size": args_cfg.get("batch", 1),
            "gradient_accumulation_steps": args_cfg.get("grad_accum", 4),
            "learning_rate": args_cfg.get("lr", 2e-5),
            "num_train_epochs": args_cfg.get("epochs", 1.0),
            "fp16": True,
            "bf16": False,
            "logging_steps": 10,
            "save_strategy": "epoch",
            "report_to": ["none"],
            # ----- critical flag ----------------------------------------------------------
            # We keep *all* original columns so that custom tokenisation performed below
            # can freely decide what to consume.  This also avoids the runtime ValueError
            # observed when the dataset columns do not match the model signature.
            "remove_unused_columns": False,
        }

        sig_params = signature(TrainingArguments.__init__).parameters
        if "evaluation_strategy" in sig_params:
            base_kwargs["evaluation_strategy"] = "epoch"

        try:
            self.tr_args = TrainingArguments(**base_kwargs)
        except TypeError as e:
            # Graceful degradation for stripped-down builds -------------------------------
            unkn_msg = str(e)
            for opt_key in ["evaluation_strategy", "remove_unused_columns"]:
                if opt_key in unkn_msg:
                    base_kwargs.pop(opt_key, None)
            self.tr_args = TrainingArguments(**base_kwargs)

        self.trainer = Trainer(
            model=model,
            args=self.tr_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            train_dataset=None,  # will be filled in train()
            eval_dataset=None,
            tokenizer=tokenizer,
        )

    # --------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------
    def _needs_tokenisation(self, ds) -> bool:
        """Return True if *ds* does not yet contain an 'input_ids' column."""
        return "input_ids" not in ds.column_names

    def _tokenise_dataset(self, ds):
        """Add `input_ids` (+ labels) columns via the stored tokenizer."""

        def _select_text_field(batch: Dict[str, List[Any]]) -> List[str]:
            # Priority: explicit 'prompt' / 'text' else first str-valued column
            if "prompt" in batch:
                return batch["prompt"]
            if "text" in batch:
                return batch["text"]
            # Fallback: detect first str column -------------------------------------------
            for k, v in batch.items():
                if isinstance(v[0], str):  # pyright: ignore[reportGeneralTypeIssues]
                    return v
            raise RuntimeError("No textual field found for tokenisation.")

        def _tok_fn(batch: Dict[str, List[Any]]):
            texts: List[str] = _select_text_field(batch)
            tokens = self.tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=self.args_cfg.get("max_length", 512),
            )
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens

        return ds.map(
            _tok_fn,
            batched=True,
            remove_columns=ds.column_names,
            desc="Tokenising dataset",
        )

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------
    def train(self, train_ds, eval_ds):
        """Run one training / evaluation loop and return the trained model."""

        # ------------------------------------------------------------------
        # Tokenise if necessary (fail-fast for unsupported schemas).
        # ------------------------------------------------------------------
        if self._needs_tokenisation(train_ds):
            train_ds = self._tokenise_dataset(train_ds)
        if self._needs_tokenisation(eval_ds):
            eval_ds = self._tokenise_dataset(eval_ds)

        # Attach to trainer --------------------------------------------------
        self.trainer.train_dataset = train_ds
        self.trainer.eval_dataset = eval_ds
        self.trainer.train()
        return self.trainer.model
