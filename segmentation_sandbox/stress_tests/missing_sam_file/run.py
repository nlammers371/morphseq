import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root)
from segmentation_sandbox.scripts.annotations.embryo_metadata import EmbryoMetadata

# Intentionally provide a non-existent SAM2 file path
try:
    metadata = EmbryoMetadata('nonexistent_sam2.json')
    print('Initialization unexpectedly succeeded')
except Exception as e:
    print('Caught exception as expected:')
    print(type(e).__name__, '-', e)
