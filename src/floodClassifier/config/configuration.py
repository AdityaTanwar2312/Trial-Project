import os
from pathlib import Path
from src.floodClassifier.constants import *
import yaml
from pathlib import Path
from typing import Any, Dict
from src.floodClassifier.entity.config_entity import BaseModelConfig, dataIngestionConfig
from src.floodClassifier.utils.common import read_yaml, create_directories

class ConfigurationManager:
    def __init__(self,
                 config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifact_root])

    def get_data_ingestion_config(self) -> dataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = dataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )

        return data_ingestion_config
    
    def get_prepare_base_model_config(self) -> Dict[str, Any]:
        pb = self._config.get("prepare_base_model", {})
        return {
            "root_dir": pb.get("root_dir", "artifacts/prepareBaseModel"),
            "base_model_path": pb.get("base_model_path", "artifacts/prepareBaseModel/base_model.pkl")
        }

    def get_arima_params(self) -> Dict[str, Any]:
        model_cfg = self._params.get("MODEL", {})
        tuning_cfg = self._params.get("TUNING", {})
        forecast_cfg = self._params.get("FORECAST", {})
        metrics = self._params.get("METRICS", [])
        return {
            "type": model_cfg.get("TYPE", "arima"),
            "p": int(model_cfg.get("P", 1)),
            "d": int(model_cfg.get("D", 0)),
            "q": int(model_cfg.get("Q", 0)),
            "seasonal": bool(model_cfg.get("SEASONAL", False)),
            "m": int(model_cfg.get("M", 0)),
            "enforce_stationarity": bool(model_cfg.get("ENFORCE_STATIONARITY", True)),
            "enforce_invertibility": bool(model_cfg.get("ENFORCE_INVERTIBILITY", True)),
            "tuning_enabled": bool(tuning_cfg.get("ENABLED", False)),
            "tuning_search": tuning_cfg.get("SEARCH", "grid"),
            "p_values": tuning_cfg.get("P_VALUES", []),
            "d_values": tuning_cfg.get("D_VALUES", []),
            "q_values": tuning_cfg.get("Q_VALUES", []),
            "horizon": int(forecast_cfg.get("HORIZON", 30)),
            "conf_int": float(forecast_cfg.get("CONF_INT", 0.95)),
            "metrics": metrics
        }