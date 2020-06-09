import os
import json5

def _load_param(file: str):
    #file = os.path.join(root, file)
    if not file.endswith('.json5'):
        file += '.json5'
    with open(file) as f:
        config = json5.load(f)
        return config