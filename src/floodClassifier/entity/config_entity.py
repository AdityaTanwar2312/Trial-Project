from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class dataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class BaseModelConfig:
    root_dir: Path
    base_model_path: Path