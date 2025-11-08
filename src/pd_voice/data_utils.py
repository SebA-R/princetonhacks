from __future__ import annotations

import hashlib
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

from .config import CONFIG, FigshareFile, TrainingConfig


@dataclass
class SampleRecord:
    sample_id: str
    filepath: Path
    label: int
    cohort: str
    duration_sec: Optional[float] = None
    age: Optional[float] = None
    sex: Optional[str] = None


def _md5(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        progress = tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {destination.name}",
        )
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))
        progress.close()


def _extract_zip(zip_path: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(destination)


def ensure_dataset(config: TrainingConfig = CONFIG) -> Dict[str, Path]:
    """Download and extract the Figshare dataset if needed."""
    assets: Dict[str, Path] = {}
    for name, file_meta in config.figshare_files.items():
        target_path = config.raw_dir / name
        if not target_path.exists() or _md5(target_path) != file_meta.md5:
            _download_file(file_meta.download_url, target_path)
        assets[name] = target_path
        if _md5(target_path) != file_meta.md5:
            raise ValueError(f"Checksum mismatch for {name}")
        if file_meta.kind == "zip":
            _extract_zip(target_path, config.raw_dir / Path(name).stem)
    return assets


def _normalize_columns(columns: Iterable[str]) -> List[str]:
    normalized = []
    for col in columns:
        col_key = (
            col.strip()
            .lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
        )
        normalized.append(col_key)
    return normalized


def load_demographics(
    demographics_path: Path,
) -> pd.DataFrame:
    if not demographics_path.exists():
        raise FileNotFoundError(f"Missing demographics spreadsheet at {demographics_path}")
    df = pd.read_excel(demographics_path)
    df.columns = _normalize_columns(df.columns)
    column_aliases = {
        "sample_id": "sample_id",
        "sample": "sample_id",
        "participant_id": "sample_id",
        "label": "label",
        "diagnosis": "label",
        "sex": "sex",
        "gender": "sex",
        "age": "age",
    }
    renamed = {}
    for col in df.columns:
        if col in column_aliases:
            renamed[col] = column_aliases[col]
    df = df.rename(columns=renamed)
    for required in ["sample_id", "label"]:
        if required not in df.columns:
            raise ValueError(f"Demographics file missing `{required}` column after renaming.")
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    df["label_id"] = df["label"].map({"HC": 0, "PWPD": 1, "PD": 1})
    return df


def collect_audio_samples(config: TrainingConfig = CONFIG) -> List[SampleRecord]:
    """Enumerate audio files with inferred labels from parent folders."""
    ensure_dataset(config)
    cohort_dirs = {"PD_AH": 1, "HC_AH": 0}
    samples: List[SampleRecord] = []
    for cohort, label in cohort_dirs.items():
        root = config.raw_dir / cohort
        if not root.exists():
            raise FileNotFoundError(f"Expected directory {root} was not created.")
        wav_files = sorted(root.rglob("*.wav"))
        for wav_path in wav_files:
            samples.append(
                SampleRecord(
                    sample_id=wav_path.stem,
                    filepath=wav_path,
                    label=label,
                    cohort=cohort,
                )
            )
    demographics = load_demographics(config.raw_dir / "Demographics_age_sex.xlsx")
    demo_map = demographics.set_index("sample_id").to_dict(orient="index")
    for sample in samples:
        if sample.sample_id in demo_map:
            entry = demo_map[sample.sample_id]
            sample.age = entry.get("age")
            sample.sex = entry.get("sex")
    return samples


def samples_to_dataframe(samples: List[SampleRecord]) -> pd.DataFrame:
    records = []
    for sample in samples:
        records.append(
            {
                "sample_id": sample.sample_id,
                "filepath": str(sample.filepath),
                "label": sample.label,
                "cohort": sample.cohort,
                "age": sample.age,
                "sex": sample.sex,
                "duration_sec": sample.duration_sec,
            }
        )
    return pd.DataFrame(records)
