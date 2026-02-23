import os, sys, json, tempfile
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root)
from segmentation_sandbox.scripts.annotations.embryo_metadata import EmbryoMetadata

sam_data = {
    "experiments": {
        "exp1": {
            "videos": {
                "vid1": {
                    "images": {
                        "exp1_vid1_t0100": {"embryos": {"exp1_e01": {}}}
                    }
                }
            }
        }
    }
}

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(sam_data, f)
    sam_path = f.name

invalid_annotations = os.path.join('nonexistent_dir', 'annotations.json')

try:
    EmbryoMetadata(sam_path, annotations_path=invalid_annotations)
    print('Initialization unexpectedly succeeded')
except Exception as e:
    print('Caught exception as expected:')
    print(type(e).__name__, '-', e)
finally:
    os.remove(sam_path)
