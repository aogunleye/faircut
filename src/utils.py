import os
import yaml
import logging

def load_config(config_path: str = "config/config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Fichier de configuration introuvable : {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def setup_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(name)