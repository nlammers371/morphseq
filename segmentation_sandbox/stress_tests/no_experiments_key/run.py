import os, sys, json, tempfile
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root)
from segmentation_sandbox.scripts.annotations.embryo_metadata import EmbryoMetadata

sam_data = {"wrong_key": {}}
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(sam_data, f)
    bad_path = f.name

try:
    EmbryoMetadata(bad_path)
    print('Initialization unexpectedly succeeded')
except Exception as e:
    print('Caught exception as expected:')
    print(type(e).__name__, '-', e)
finally:
    os.remove(bad_path)
