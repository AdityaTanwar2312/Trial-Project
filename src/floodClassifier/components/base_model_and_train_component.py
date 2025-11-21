import os
import pickle
from pathlib import Path
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.floodClassifier.entity.config_entity import BaseModelConfig
from typing import Optional, Dict
from src.floodClassifier.constants import *

class PrepareBaseModel:
    def __init__(self, arima_params: Dict[str, any], config: BaseModelConfig):
        self.config = config
        self.root_dir = Path(config.root_dir)
        self.base_model_path = Path(config.base_model_path)
        self.arima_params = arima_params
        os.makedirs(self.root_dir, exist_ok=True)

    def _ensure_series(self, y: pd.Series) -> pd.Series:
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        if y.index.dtype == object:
            try:
                y.index = pd.to_datetime(y.index)
            except Exception:
                pass
        return y.astype(float)

    def build_arima(self, y: pd.Series, exog: Optional[pd.DataFrame] = None):
        y = self._ensure_series(y)
        p = self.arima_params["p"]
        d = self.arima_params["d"]
        q = self.arima_params["q"]
        seasonal = self.arima_params["seasonal"]
        m = self.arima_params["m"] if seasonal else 0

        model = SARIMAX(
            endog=y,
            exog=exog,
            order=(p, d, q),
            seasonal_order=(0, 0, 0, 0) if not seasonal else (p, d, q, m),
            enforce_stationarity=self.arima_params.get("enforce_stationarity", True),
            enforce_invertibility=self.arima_params.get("enforce_invertibility", True),
            simple_differencing=False
        )
        return model

    def fit_and_save(self, y: pd.Series, exog: Optional[pd.DataFrame] = None, save_overwrite: bool = True):
        model = self.build_arima(y, exog=exog)
        fitted = model.fit(disp=False)
        if self.base_model_path.exists() and not save_overwrite:
            raise FileExistsError(f"Base model already exists at {self.base_model_path}")
        with open(self.base_model_path, "wb") as f:
            pickle.dump(fitted, f)
        return fitted

    def load(self):
        if not self.base_model_path.exists():
            raise FileNotFoundError(f"No base model at {self.base_model_path}")
        with open(self.base_model_path, "rb") as f:
            return pickle.load(f)

    def run_from_df(self, df: pd.DataFrame, target_col: str = "y", exog_cols: Optional[list] = None):
        if target_col not in df.columns:
            raise KeyError(f"target_col '{target_col}' not found in dataframe")
        y = df[target_col]
        exog = df[exog_cols] if exog_cols else None
        fitted = self.fit_and_save(y, exog=exog)
        return fitted
