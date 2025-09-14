# src/evaluate.py
"""Evaluation, metrics and plotting helpers (iteration-14).

The file keeps the optional *vLLM* wrapper but now points the default plot
output directory to `.research/iteration14/images` in order to comply with the
mandatory save-path constraints.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Optional vLLM import – CPU-only environments fall back to a lightweight stub
# so the rest of the codebase can still be imported.
# ---------------------------------------------------------------------------
try:
    import vllm  # noqa: F401 – optional dependency

    LLM = vllm.LLM
    SamplingParams = vllm.SamplingParams  # attribute exists in vLLM
except ModuleNotFoundError:  # pragma: no cover – executed only when vllm missing
    vllm = None
    LLM = None

    class _SamplingParams:  # noqa: D401
        """Minimal stub so that static analysers and runtime both stay happy."""

        def __init__(self, *_: Any, **__: Any):  # noqa: D401
            pass

    SamplingParams = _SamplingParams  # alias for uniform API

__all__ = [
    "Evaluator",
    "Plotter",
]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
class Evaluator:
    """Thin wrapper around vLLM for latency and ASR evaluation."""

    def __init__(self, model_id: str, *, tensor_parallel: int = 2):
        if LLM is None:
            raise RuntimeError(
                "`vllm` is not installed or no compatible GPU was detected.  "
                "Install vLLM and ensure that a CUDA device is available, or "
                "disable evaluation that depends on it."
            )
        self.llm = LLM(model=model_id, tensor_parallel_size=tensor_parallel)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def single_pass(
        self,
        prompts: List[str],
        *,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1024,
    ) -> List[str]:
        sp = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
        outs = self.llm.generate(prompts, sp)
        return [o.outputs[0].text for o in outs]

    # ------------------------------------------------------------------
    def benchmark_latency(self, prompt: str, *, n: int = 10, **decode_kwargs):
        lat = []
        for _ in range(n):
            t0 = time.perf_counter()
            _ = self.single_pass([prompt], **decode_kwargs)
            lat.append(time.perf_counter() - t0)
        return float(np.mean(lat)), float(np.std(lat))

    # ------------------------------------------------------------------
    def compute_asr(self, prompts: List[str], bad_keywords: List[str], **decode_kwargs):
        """Coarse automatic metric: mark success if any *bad_keyword* appears."""

        generations = self.single_pass(prompts, **decode_kwargs)
        succ = 0
        for g in generations:
            if any(kw.lower() in g.lower() for kw in bad_keywords):
                succ += 1
        return succ / len(prompts)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


class Plotter:
    """Create bar plots saved under `.research/iteration14/images`."""

    def __init__(self, out_dir: Path | None = None):
        # default directory complies with mandatory iteration-14 path.
        self.out_dir = out_dir or Path(".research/iteration14/images")
        self.out_dir.mkdir(exist_ok=True, parents=True)

    # ------------------------------------------------------------------
    def bar(
        self,
        values: Dict[str, float],
        *,
        title: str,
        fname: str,
        ylabel: str = "ASR (%)",
    ) -> str:
        labels = list(values.keys())
        vals = [v * 100 for v in values.values()]
        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(x, vals, color="skyblue")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        for b, v in zip(bars, vals, strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.5,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        pdf_path = self.out_dir / fname
        fig.tight_layout()
        fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
        plt.close(fig)
        return pdf_path.name
