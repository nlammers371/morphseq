import os, sys, tempfile
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root)
from segmentation_sandbox.scripts.annotations.embryo_metadata import EmbryoMetadata

# Create a temporary file with invalid JSON content
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    f.write('{ invalid json }')
    bad_path = f.name

try:
    EmbryoMetadata(bad_path)
    print('Initialization unexpectedly succeeded')
except Exception as e:
    print('Caught exception as expected:')
    print(type(e).__name__, '-', e)
finally:
    os.remove(bad_path)
