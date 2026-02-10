import tempfile
from pathlib import Path
import os
import shutil

from scripts.utils.parsing_utils import get_image_filename_from_id, build_image_id
from scripts.data_organization.data_organizer import DataOrganizer
from scripts.detection_segmentation.sam2_utils import run_sam2_propagation


def test_parsing_utils_full_filename():
    image_id = '20250624_chem02_28C_T00_1356_H01_t0042'
    filename = get_image_filename_from_id(image_id)
    assert filename == f'{image_id}.jpg'


def test_data_organizer_get_image_path_from_id():
    image_id = '20250624_chem02_28C_T00_1356_H01_t0042'
    with tempfile.TemporaryDirectory() as td:
        images_dir = Path(td)
        expected = images_dir / f'{image_id}.jpg'
        assert DataOrganizer.get_image_path_from_id(image_id, images_dir) == expected


class DummyPredictor:
    def init_state(self, video_path: str):
        # Just return a dummy state that indicates success
        return {'video_path': video_path}


def test_run_sam2_propagation_symlink_creation():
    # Create a small set of fake images named by image_id
    video_id = '20250624_chem02_28C_T00_1356_H01'
    image_ids = [build_image_id(video_id, i) for i in range(3)]
    with tempfile.TemporaryDirectory() as td:
        images_dir = Path(td)
        # create fake image files
        for image_id in image_ids:
            (images_dir / f'{image_id}.jpg').write_text('fake')

        # Create a dummy predictor and call run_sam2_propagation expecting no FileNotFoundError
        predictor = DummyPredictor()
        # seed_detections and embryo ids can be minimal
        seed_detections = [{'box_xyxy': [0,0,10,10]}]
        embryo_ids = ['20250624_chem02_28C_T00_1356_H01_e01']

        # run_sam2_propagation expects a video_dir Path; pass images_dir
        try:
            result = run_sam2_propagation(predictor, images_dir, 0, seed_detections, embryo_ids, image_ids, verbose=False)
        except FileNotFoundError:
            assert False, 'run_sam2_propagation failed to find created image files'

        # If no exception, test passed
        assert True
