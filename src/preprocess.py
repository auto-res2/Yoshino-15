# src/preprocess.py
"""Dataset download / caching utilities."""
from __future__ import annotations

import hashlib
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Dict

import requests
import tqdm
from datasets import Dataset, load_dataset
from huggingface_hub import login

# ---------------------------------------------------------------------------
# Authenticate to HF Hub if token is provided as environment variable.
# ---------------------------------------------------------------------------
_HF_TOKEN = os.getenv("HF_TOKEN", None)
if _HF_TOKEN:
    login(token=_HF_TOKEN)


class DataManager:
    """Resolve dataset specifications from YAML into 🤗 Datasets objects."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    # ------------------------------------------------------------------
    def _download_url(self, url: str, dst: Path):
        """Stream download a remote file with a progress bar."""

        if dst.exists():
            return dst

        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dst, "wb") as f, tqdm.tqdm(total=total, unit="B", unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return dst

    # ------------------------------------------------------------------
    @staticmethod
    def _unpack(file_path: Path, dst_dir: Path):
        if file_path.suffix == ".zip":
            with zipfile.ZipFile(file_path, "r") as zf:
                zf.extractall(dst_dir)
        elif file_path.suffix in {".tar", ".gz", ".tgz", ".tar.gz"}:
            with tarfile.open(file_path, "r:*") as tf:
                tf.extractall(dst_dir)

    # ------------------------------------------------------------------
    def resolve_dataset(self, spec: Dict, *, split: str = "train", smoke: bool = False) -> Dataset:
        """Convert a *spec* entry from YAML into a 🤗 Dataset."""

        if "hf_repo" in spec:
            repo = spec["hf_repo"]
            cfg = spec.get("config", None)
            ds = load_dataset(repo, name=cfg, split=split, cache_dir=str(self.cache_dir))
            if smoke:
                ds = ds.select(range(min(50, len(ds))))
            return ds

        if "url" in spec:
            url = spec["url"]
            sha = spec.get("sha256")
            fname = Path(url).name
            downloaded = self._download_url(url, self.cache_dir / fname)

            # ----------------------------------------------------------
            # SHA-256 integrity check
            # ----------------------------------------------------------
            if sha:
                h = hashlib.sha256(downloaded.read_bytes()).hexdigest()
                if h != sha:
                    raise ValueError(f"SHA256 mismatch for {url}")

            extract_dir = self.cache_dir / f"{downloaded.stem}_extract"
            extract_dir.mkdir(exist_ok=True, parents=True)
            self._unpack(downloaded, extract_dir)

            jsonl_files = list(extract_dir.rglob("*.jsonl"))
            if not jsonl_files:
                raise RuntimeError(f"No .jsonl found inside {extract_dir}")

            data_files = {"data": [str(f) for f in jsonl_files]}
            ds = load_dataset("json", data_files=data_files, split="data")
            if smoke:
                ds = ds.select(range(min(50, len(ds))))
            return ds

        raise KeyError("Dataset spec must contain either 'hf_repo' or 'url'.")
