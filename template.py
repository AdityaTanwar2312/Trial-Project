import os
from pathlib import Path
import logging

# Logging stream

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: [%(message)s]') # Timestamp and message

project_name = 'cnnClassifier'

list_of_files = [
    ".github/workflows/.gitkeep", # GitHub workflows directory
    f"src/{project_name}/__init__.py", # Source directory with project name
    f"src/{project_name}/components/__init__.py", # Components subdirectory
    f"src/{project_name}/utils/__init__.py", # Utils subdirectory
    f"src/{project_name}/config/__init__.py", # Config subdirectory
    f"src/{project_name}/config/configuration.py", # Configuration file
    f"src/{project_name}/pipeline/__init__.py", # Pipeline subdirectory
    f"src/{project_name}/entity/__init__.py", # Entities subdirectory
    f"src/{project_name}/constants/__init__.py", # Constants subdirectory
    "config/config.yaml", # Configuration YAML file
    "dvc.yaml", # DVC pipeline file
    "params.yaml", # Parameters YAML file
    "requirements.txt", # Requirements file
    "setup.py", # Setup script
    "research/trials.ipynb", # Research notebook
    "templates/index.html" # HTML template file
]

for filename in list_of_files:
    filepath = Path(filename) # Create a Path object
    filedir, file_name = os.path.split(filepath) # Split into directory and file name

    if filedir != "":
        os.makedirs(filedir, exist_ok=True) # Create directories if they don't exist
        logging.info(f"Created directory: {filedir} for file: {file_name}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0): # Check if file doesn't exist or is empty
        with open(filepath, 'w') as fp:
            pass # Create an empty file
        logging.info(f"Created empty file: {filepath}")
    else:
        logging.info(f"File already exists and is not empty: {filepath}")