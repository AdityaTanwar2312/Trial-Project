from src.floodClassifier.config.configuration import ConfigurationManager
from src.floodClassifier.components.base_model_and_train_component import PrepareBaseModel
from src.floodClassifier import logger
import pandas as pd

STAGE_NAME = 'Base Model and Training'

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            prepare_cfg = config.get_prepare_base_model_config()
            arima_params = config.get_arima_params()
            preparer = PrepareBaseModel(prepare_cfg, arima_params)

            csv_path = r"artifacts\data_ingestion\FloodPrediction.csv"

            date_year_col = "Year"
            date_month_col = "Month"
            target_col = "Flood?"
            exog_cols = ["Rainfall", "Max_Temp", "Min_Temp", "Relative_Humidity"]
            usecols = [date_year_col, date_month_col, target_col] + exog_cols
            df = pd.read_csv(csv_path, usecols=usecols)

            try:
                df["Month"] = df[date_month_col].astype(int)
                df["Year"] = df[date_year_col].astype(int)
                df["_date"] = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=1))
            except Exception:
                df["_date"] = pd.to_datetime(df[date_year_col].astype(str) + "-" + df[date_month_col].astype(str) + "-01", errors="coerce")

            df = df.dropna(subset=["_date"]).copy()
            df = df.set_index("_date")
            df = df.sort_index()

            if target_col not in df.columns:
                raise KeyError(f"Target column '{target_col}' not found in file: {csv_path}")

            df = df.dropna(subset=[target_col])

            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
            if df[target_col].isna().any():
                raise ValueError("Target column contains non-numeric or uncoercible values after parsing")

            if exog_cols:
                for col in exog_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df[exog_cols] = df[exog_cols].fillna(method="ffill").fillna(method="bfill")

            if not df.index.is_monotonic_increasing:
                df = df.sort_index()

            required_obs = max(arima_params.get("p", 1), arima_params.get("d", 0), arima_params.get("q", 1)) + 1
            if len(df) < required_obs:
                raise ValueError(f"Not enough observations ({len(df)}) to fit ARIMA with order p,d,q requiring at least {required_obs}")

            fitted_model = preparer.run_from_df(df, target_col=target_col, exog_cols=exog_cols)
            logger.info(f"ARIMA base model fitted and saved to: {prepare_cfg['base_model_path']}")

        except Exception as e:
            raise e
        
if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e