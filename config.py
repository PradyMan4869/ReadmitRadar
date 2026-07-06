"""Central configuration: paths and environment-driven settings."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
DATA_DIR = ROOT / "storage"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(exist_ok=True)


@dataclass(frozen=True)
class LMStudioConfig:
    base_url: str = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    model: str = os.getenv("LMSTUDIO_MODEL", "mistral-7b-instruct-v0.3")
    timeout_s: int = 60


@dataclass(frozen=True)
class LangfuseConfig:
    host: str = os.getenv("LANGFUSE_HOST", "")
    public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "")

    @property
    def enabled(self) -> bool:
        return bool(self.public_key and self.secret_key)


@dataclass(frozen=True)
class TrainConfig:
    n_hospital_a: int = 6000
    n_hospital_b: int = 6000
    n_test: int = 3000
    fedavg_rounds: int = 10
    random_state: int = 42


LMSTUDIO = LMStudioConfig()
LANGFUSE = LangfuseConfig()
TRAIN = TrainConfig()
