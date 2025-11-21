import zipfile
import os
import gdown
from pathlib import Path
from src.floodClassifier import logger
from src.floodClassifier.entity.config_entity import dataIngestionConfig

class DataIngestion:
    def __init__(self, config: dataIngestionConfig):
        self.config = config

    def download_file(self) -> Path:
        if os.path.exists(self.config.local_data_file):
            logger.info(f"File already exists at {self.config.local_data_file}. Skipping download.")
        else:
            logger.info(f"Downloading file from {self.config.source_URL} to {self.config.local_data_file}")
            gdown.download(url=self.config.source_URL, output=str(self.config.local_data_file), quiet=False)
            logger.info(f"File downloaded successfully to {self.config.local_data_file}")
        return self.config.local_data_file

    def extract_zip_file(self, zip_file_path: Path):
        logger.info(f"Extracting zip file {zip_file_path} to directory {self.config.unzip_dir}")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.config.unzip_dir)
        logger.info(f"Extraction completed successfully to directory {self.config.unzip_dir}")

    def initiate_data_ingestion(self):
        zip_file_path = self.download_file()
        self.extract_zip_file(zip_file_path)
        return self.config.unzip_dir