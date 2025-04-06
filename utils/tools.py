import os

def create_cache_dir() -> None:
    """
    Create a cache directory for storing temporary files.
    """
    cache_dir = os.path.join(os.getcwd(), "cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    