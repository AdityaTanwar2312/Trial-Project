import os
import box.exceptions as BoxValueError
import yaml
import base64
import json
import joblib
from src.floodClassifier import logger
from ensure import ensure_annotations
from box import ConfigBox
from typing import Any
from pathlib import Path
from typing import Union

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a yaml file and returns

    Args:
        path_to_yaml (Path): Path to the yaml file

    Raises:
        e: EmptyFileError if yaml file is empty
        e: YAMLError if yaml file is malformed

    Returns:
        ConfigBox: ConfigBox type object
    """
    try:
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError as e:
        raise BoxValueError("EmptyFileError: yaml file is empty")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"yaml file {path_to_yaml} is malformed")
    except Exception as e:
        raise e
    
@ensure_annotations
def write_yaml(path_to_yaml: Path, content: Any, replace: bool = False) -> None:
    """Writes content to a yaml file

    Args:
        path_to_yaml (Path): Path to the yaml file
        content (Any): Content to be written to the yaml file
        replace (bool, optional): Whether to replace the existing file. Defaults to False.

    Raises:
        e: YAMLError if yaml file is malformed
    """
    try:
        if path_to_yaml.exists() and replace is False:
            logger.info(f"yaml file: {path_to_yaml} already exists. Not overwriting.")
            return
        with open(path_to_yaml, 'w') as yaml_file:
            yaml.safe_dump(content, yaml_file)
            logger.info(f"yaml file: {path_to_yaml} written successfully")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"yaml file {path_to_yaml} is malformed")
    except Exception as e:
        raise e
    
def create_directories(path_to_directories: list, verbose: bool = True) -> None:
    """Creates list of directories

    Args:
        path_to_directories (list): List of directory paths
        verbose (bool, optional): Whether to log the directory creation. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory created at: {path}")


def save_json(path_to_json: Path, data: object, replace: bool = False) -> None:
    try:
        if path_to_json.exists() and replace is False:
            logger.info(f"json file: {path_to_json} already exists. Not overwriting.")
            return
        with open(path_to_json, 'w') as json_file:
            json.dump(data, json_file, indent=4)
            logger.info(f"json file: {path_to_json} saved successfully")
    except Exception as e:
        raise e
    
@ensure_annotations
def load_json(path_to_json: Path) -> Any:
    try:
        with open(path_to_json, 'r') as json_file:
            data = json.load(json_file)
            logger.info(f"json file: {path_to_json} loaded successfully")
            return data
    except Exception as e:
        raise e

@ensure_annotations
def save_bin(data: Any, path_to_bin: Path) -> None:
    try:
        joblib.dump(data, path_to_bin)
        logger.info(f"Binary file: {path_to_bin} saved successfully")
    except Exception as e:
        raise e

@ensure_annotations
def load_bin(path_to_bin: Path) -> Any:
    try:
        data = joblib.load(path_to_bin)
        logger.info(f"Binary file: {path_to_bin} loaded successfully")
        return data
    except Exception as e:
        raise e
    
@ensure_annotations
def encode_image_to_base64(image_path: Path) -> str:
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            logger.info(f"Image at {image_path} encoded to base64 successfully")
            return encoded_string
    except Exception as e:
        raise e

@ensure_annotations
def decode_base64_to_image(base64_string: str, output_image_path: Path) -> None:
    try:
        image_data = base64.b64decode(base64_string)
        with open(output_image_path, "wb") as image_file:
            image_file.write(image_data)
            logger.info(f"Base64 string decoded to image at {output_image_path} successfully")
    except Exception as e:
        raise e