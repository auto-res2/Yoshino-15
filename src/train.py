# src/train.py
"""Model construction and fine-tuning utilities (iteration-10)."""
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
# a stripped-down variant that lacks certain fields.  We therefore detect the
# available signature at runtime and only forward the supported kwargs—while
# keeping a strict fail-fast stance for all unrelated errors.
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

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_base(self, model_id: str, *, quant: bool = False):
        """Load a pretrained causal-LM in float16 or 8-bit."""

        bnb_cfg = None
        if quant:
            # bitsandbytes 8-bit quantisation
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["lm_head"])

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,  # ``torch_dtype`` deprecated ≥4.41
            device_map="auto",
            quantization_config=bnb_cfg,
            cache_dir=self.cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=self.cache_dir)
        # ------------------------------------------------------------------
        # Ensure padding token exists so that the DataCollator can work.
        # ------------------------------------------------------------------
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    # ------------------------------------------------------------------
    # LoRA helpers
    # ------------------------------------------------------------------
    @staticmethod
    def add_lora(model, *, r: int = 16, alpha: int = 32, dropout: float = 0.05):
        """Wrap *model* with PEFT / LoRA adapters."""

        lora_cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=dropout,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        return model


class TrainerWrapper:
    """Light wrapper around 🤗 Trainer to keep *main.py* readable."""

    def __init__(self, model, tokenizer, *, output_dir: Path, args_cfg: Dict[str, Any]):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.args_cfg = args_cfg  # keep for later helper usage
        self.tokenizer = tokenizer

        # ------------------------------------------------------------------
        # Dynamically build a kwargs dict limited to the parameters actually
        # supported by the detected ``TrainingArguments`` implementation.
        # ------------------------------------------------------------------
        base_kwargs: Dict[str, Any] = {
            "output_dir": str(self.output_dir),
            "per_device_train_batch_size": int(args_cfg.get("batch", 1)),
            "gradient_accumulation_steps": int(args_cfg.get("grad_accum", 4)),
            # Cast to float explicitly – YAML may treat scientific notation as str
            "learning_rate": float(args_cfg.get("lr", 2e-5)),
            "num_train_epochs": float(args_cfg.get("epochs", 1.0)),
            "fp16": True,
            "bf16": False,
            "logging_steps": 10,
            "save_strategy": "epoch",
            "report_to": ["none"],
            # Keep all columns so our custom tokenisation can access arbitrary fields
            "remove_unused_columns": False,
        }

        sig_params = signature(TrainingArguments.__init__).parameters
        if "evaluation_strategy" in sig_params:
            base_kwargs["evaluation_strategy"] = "epoch"

        try:
            self.tr_args = TrainingArguments(**base_kwargs)
        except TypeError as e:
            # Graceful degradation for stripped-down builds; remove unsupported keys
            err_msg = str(e)
            for opt_key in ["evaluation_strategy", "remove_unused_columns"]:
                if opt_key in err_msg:
                    base_kwargs.pop(opt_key, None)
            self.tr_args = TrainingArguments(**base_kwargs)

        # ------------------------------------------------------------------
        # Extra safety: make 100 % sure the LR is *float* before the optimiser
        # is constructed.  Some rare edge-cases (YAML parsing quirks, env vars)
        # may still sneak in a string at this point.
        # ------------------------------------------------------------------
        self.tr_args.learning_rate = float(self.tr_args.learning_rate)

        self.trainer = Trainer(
            model=model,
            args=self.tr_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            train_dataset=None,  # filled later
            eval_dataset=None,
            tokenizer=tokenizer,
        )

    # --------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------
    @staticmethod
    def _needs_tokenisation(ds) -> bool:
        """Return *True* if *ds* lacks an ``input_ids`` column."""
        return "input_ids" not in ds.column_names

    def _tokenise_dataset(self, ds):
        """Add ``input_ids`` (and *let the DataCollator create labels*) columns via the stored tokenizer."""

        def _select_text_field(batch: Dict[str, List[Any]]) -> List[str]:
            if "prompt" in batch:
                return batch["prompt"]
            if "text" in batch:
                return batch["text"]
            # Fallback – choose the first string-typed column encountered
            for _key, value in batch.items():
                if isinstance(value[0], str):
                    return value  # runtime type guarantee
            raise RuntimeError("No textual field found for tokenisation.")

        def _tok_fn(batch: Dict[str, List[Any]]):
            texts: List[str] = _select_text_field(batch)
            tokens = self.tokenizer(
                texts,
                truncation=True,
                padding=False,  # DataCollator will pad dynamically per batch
                max_length=int(self.args_cfg.get("max_length", 512)),
                return_attention_mask=True,
            )
            # Do *not* add "labels" here – DataCollatorForLanguageModeling will create them
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

        if self._needs_tokenisation(train_ds):
            train_ds = self._tokenise_dataset(train_ds)
        if self._needs_tokenisation(eval_ds):
            eval_ds = self._tokenise_dataset(eval_ds)

        # Strap datasets into trainer and go
        self.trainer.train_dataset = train_ds
        self.trainer.eval_dataset = eval_ds
        self.trainer.train()
        return self.trainer.model
