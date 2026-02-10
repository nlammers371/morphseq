segmentation_sandbox/
├── configs/
│   └── pipeline_config.yaml
├── data/
│   ├── annotation_and_masks/
│   │   ├── gdino_annotations/
│   │   │   ├── gdino_annotations.json
│   │   │   ├── gdino_annotations_finetuned.json
│   │   │   └── video_mode_changes.csv
│   │   └── jpg_masks/
│   │       └── 20240404/
│   │           └── masks/
│   │               └── 20240404_E05_0011_masks_emnum_1.jpg
│   └── raw_data_organized/
│       ├── experiment_metadata.json
│       └── [other experiment folders/files]
├── scripts/
│   ├── 03_gdino_detection_with_filtering.py
│   ├── tests/
│   │   └── understand_dino_distrbiution.ipynb
│   ├── utils/
│   │   ├── experiment_metadata_utils.py
│   │   ├── grounded_sam_utils.py
│   │   └── demos/
│   │       ├── gdino_side_by_side_videos.py
│   │       └── video_mode_changes.py
└── [other files/folders as present]