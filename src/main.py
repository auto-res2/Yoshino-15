# src/main.py
"""Orchestrate smoke-test and full experiments from the command line.

Usage:
    uv run python -m src.main --smoke-test
    uv run python -m src.main --full-experiment
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

from .preprocess import DataManager
from .train import ModelBuilder, TrainerWrapper
from .evaluate import Evaluator, Plotter

# ---------------------------------------------------------------------------
# Project-level paths  (UPDATED to iteration6 as per mandatory requirement)
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESEARCH_DIR = PROJECT_DIR / ".research" / "iteration6"
IMAGES_DIR = RESEARCH_DIR / "images"
RESULTS_DIR = RESEARCH_DIR  # JSON lives directly inside iteration6/
DATA_DIR = PROJECT_DIR / "data"

# Ensure directories exist ---------------------------------------------------
for p in (IMAGES_DIR, RESULTS_DIR, DATA_DIR):
    p.mkdir(exist_ok=True, parents=True)

# ---------------------------------------------------------------------------
# Configuration helper
# ---------------------------------------------------------------------------
CONFIG_DIR = PROJECT_DIR / "config"


@dataclass
class ExperimentConfig:
    name: str
    description: str
    datasets: Dict[str, Any]
    models: Dict[str, Any]
    training: Dict[str, Any]
    evaluation: Dict[str, Any]
    plotting: Dict[str, Any]

    @staticmethod
    def from_file(path: Path) -> "ExperimentConfig":
        with path.open("r") as f:
            raw = yaml.safe_load(f)
        return ExperimentConfig(**raw)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def set_seed(sd: int):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)


# ---------------------------------------------------------------------------
# Core experiment driver
# ---------------------------------------------------------------------------

def run_experiment(cfg: ExperimentConfig, *, smoke: bool):
    results: Dict[str, Any] = {
        "experiment": cfg.name,
        "description": cfg.description,
        "seed_results": [],
    }

    # helpers ---------------------------------------------------------------
    dm = DataManager(DATA_DIR)
    mb = ModelBuilder(DATA_DIR)
    plotter = Plotter(IMAGES_DIR)

    # ----------------------------------------------------------------------
    # Iterate over random seeds
    # ----------------------------------------------------------------------
    for seed in cfg.training.get("seeds", [42]):
        set_seed(seed)
        seed_rec: Dict[str, Any] = {"seed": seed, "model": {}}

        # --------------------------------------------------------------
        # For every model specified in YAML
        # --------------------------------------------------------------
        for mname, mspec in cfg.models.items():
            base_id = mspec["hf_id"]
            model, tok = mb.load_base(base_id, quant=mspec.get("8bit", False))
            if mspec.get("lora", False):
                model = mb.add_lora(model, r=mspec.get("lora_r", 16))

            # ------------------------------------------------------
            # Optional fine-tuning
            # ------------------------------------------------------
            checkpoint_dir = None  # will be set if training occurs
            if mspec.get("training") is not None:
                train_ds = dm.resolve_dataset(
                    cfg.datasets[mspec["training"]], split="train", smoke=smoke
                )
                eval_ds = dm.resolve_dataset(
                    cfg.datasets[mspec.get("eval", mspec["training"])],
                    split=["validation", "test"],  # try validation, fall back to test
                    smoke=smoke,
                )
                tw = TrainerWrapper(
                    model,
                    tok,
                    output_dir=RESULTS_DIR / "checkpoints" / mname,
                    args_cfg=cfg.training,
                )
                model = tw.train(train_ds, eval_ds)

                checkpoint_dir = RESULTS_DIR / "checkpoints" / f"{mname}_seed{seed}"
                model.save_pretrained(checkpoint_dir)

            # ------------------------------------------------------
            # Evaluation (prompt-level ASR)
            # ------------------------------------------------------
            model_path_for_eval = str(checkpoint_dir) if checkpoint_dir else base_id
            evaluator = Evaluator(model_id=model_path_for_eval)  # vLLM expects ID / local path

            attack_split = cfg.evaluation.get("attack_split", "test")
            attack_ds = dm.resolve_dataset(
                cfg.datasets[cfg.evaluation["attack_set"]], split=attack_split, smoke=smoke
            )
            column_name = (
                "prompt" if "prompt" in attack_ds.column_names else attack_ds.column_names[0]
            )
            prompts: List[str] = attack_ds[column_name]

            asr = evaluator.compute_asr(
                prompts,
                bad_keywords=cfg.evaluation.get("bad_keywords", ["harm"]),
                temperature=cfg.evaluation.get("temperature", 0.8),
                top_p=cfg.evaluation.get("top_p", 0.95),
            )
            seed_rec["model"][mname] = {"ASR": asr}

        results["seed_results"].append(seed_rec)

    # ----------------------------------------------------------------------
    # Aggregate over seeds
    # ----------------------------------------------------------------------
    agg = {}
    for m in cfg.models.keys():
        vals = [sr["model"][m]["ASR"] for sr in results["seed_results"]]
        agg[m] = float(np.mean(vals))
    results["aggregate"] = agg

    # ----------------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------------
    fig_name = plotter.bar(agg, title="Attack Success Rate (↓)", fname="asr_prompt_level.pdf")
    results["figures"] = [fig_name]

    # ----------------------------------------------------------------------
    # Persist JSON into .research/iteration6 and also print to stdout
    # ----------------------------------------------------------------------
    out_path = RESULTS_DIR / f"{cfg.name.replace(' ', '_')}_results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    print("\n================ Experiment Description ================")
    print(cfg.description)
    print("================ Numerical Results ====================")
    print(json.dumps(results, indent=2))
    print("================ Figure Files =========================")
    print("\n".join(results["figures"]))


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Run PRANCE experiments")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run the smoke-test configuration")
    group.add_argument("--full-experiment", action="store_true", help="Run the full experiment")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.smoke_test:
        cfg_path = CONFIG_DIR / "smoke_test.yaml"
        smoke = True
    else:
        cfg_path = CONFIG_DIR / "full_experiment.yaml"
        smoke = False

    if not cfg_path.exists():
        print(f"[ERROR] Configuration file not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    cfg = ExperimentConfig.from_file(cfg_path)
    run_experiment(cfg, smoke=smoke)


if __name__ == "__main__":
    main()
