from graphrag.config.load_config import load_config
from pathlib import Path
from graphrag.api.index import build_index
from grooagents.utils.tools import load_config as lymlconfig
import os
import yaml

async def index_kg():
    conf = load_config(Path("./"))
    await build_index(config=conf)

def update_input_config(kwargs: dict):
    key, val = next(iter(kwargs.items()))
    cwd = os.getcwd()
    config = lymlconfig(config_path=os.path.join(cwd, 'cache', 'kbgraph', 'settings.yaml'))

    if key == 'file_type' and val in ['text', 'str', 'json']:
        config['input']['file_type'] = val
    if key == 'base_dir':
        config['input']['base_dir'] = val

    with open(os.path.join(cwd, 'settings.yaml'), 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)

    

    