from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class FigshareFile:
    """Metadata describing a Figshare asset we need to download."""

    name: str
    download_url: str
    md5: str
    kind: str  # "zip" or "spreadsheet"


@dataclass
class TrainingConfig:
    """Centralized configuration for the Parkinson voice pipeline."""

    base_dir: Path = Path(".")
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    processed_audio_dir: Path = Path("data/processed")
    cache_dir: Path = Path(".cache/pd_voice")
    article_id: int = 23849127
    sample_rate: int = 16_000
    min_duration_sec: float = 3.0
    max_duration_sec: float = 8.0
    augmentation_per_clip: int = 2
    random_seed: int = 42
    cv_folds: int = 5
    test_size: float = 0.2
    top_db_trim: int = 35
    target_dbfs: float = -20.0
    ssl_models: List[str] = field(
        default_factory=lambda: [
            "facebook/wav2vec2-base",
            "microsoft/wavlm-base-plus",
        ]
    )

    def __post_init__(self) -> None:
        self.data_dir = (self.base_dir / self.data_dir).resolve()
        self.artifacts_dir = (self.base_dir / self.artifacts_dir).resolve()
        self.processed_audio_dir = (self.base_dir / self.processed_audio_dir).resolve()
        self.cache_dir = (self.base_dir / self.cache_dir).resolve()
        for directory in [
            self.data_dir,
            self.artifacts_dir,
            self.processed_audio_dir,
            self.cache_dir,
            self.raw_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def figshare_files(self) -> Dict[str, FigshareFile]:
        return {
            "Demographics_age_sex.xlsx": FigshareFile(
                name="Demographics_age_sex.xlsx",
                download_url="https://ndownloader.figshare.com/files/41836707",
                md5="15bfde0fcb441c55dcd2fc3138c3a672",
                kind="spreadsheet",
            ),
            "PD_AH.zip": FigshareFile(
                name="PD_AH.zip",
                download_url="https://ndownloader.figshare.com/files/41836710",
                md5="2e525e419abdad509e5cbf201e30ef36",
                kind="zip",
            ),
            "HC_AH.zip": FigshareFile(
                name="HC_AH.zip",
                download_url="https://ndownloader.figshare.com/files/41836713",
                md5="55fd33d365380372dcfbae4f7f520246",
                kind="zip",
            ),
        }


CONFIG = TrainingConfig()

