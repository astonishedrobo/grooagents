import os
import yaml

def create_cache_dir() -> None:
    """
    Create a cache directory for storing temporary files.
    """
    cache_dir = os.path.join(os.getcwd(), "cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
def load_config(config_path: str = None) -> dict:
    """
    Load a YAML configuration file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: Configuration parameters as a dictionary.
    """
    if config_path is None:
        config_path = os.path.join(os.getcwd(), "config.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
