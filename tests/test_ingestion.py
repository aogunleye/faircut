import os
from src.utils import load_config

def test_config_loading():
    config = load_config("config/config.yaml")
    assert "data" in config
    assert "tmdb" in config