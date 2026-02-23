# Segmentation Sandbox Documentation

## Pipeline Philosophy

The segmentation sandbox is built on a set of core principles designed to ensure robustness, reproducibility, and ease of use. Understanding these principles is key to effectively using and extending the pipeline.

1.  **Modularity and Separation of Concerns:** The pipeline is divided into distinct, sequential steps (e.g., data organization, detection, segmentation). Each step is a self-contained script with a clear purpose, taking a well-defined input and producing a predictable output. This modular design makes the pipeline easier to debug, test, and modify. Individual steps can be run independently, allowing for greater flexibility in the workflow.

2.  **Centralized and Strict ID Management:** The entire pipeline revolves around a unified and strictly enforced system for identifying and tracking entities (experiments, videos, images, embryos, and snips). The `parsing_utils.py` module serves as the single source of truth for all ID parsing and generation logic. This rigid adherence to a consistent naming convention is crucial for data integrity, preventing ambiguity, and enabling reliable automation and data querying.

3.  **Data-Driven and Explicit Metadata:** The pipeline's state and the flow of information between steps are managed through explicit metadata files (e.g., `experiment_metadata.json`, `gdino_annotations.json`). This approach avoids reliance on implicit information (like file paths) and makes the entire process more transparent and auditable. The `EntityIDTracker`, embedded within the metadata files, provides a comprehensive record of all entities processed by the pipeline.

4.  **Progressive Refinement of Data:** The pipeline follows a clear data flow, starting with raw images and progressively enriching them with more information at each step. Raw images are first organized, then annotated with bounding boxes, then segmented to generate masks, and finally curated with detailed metadata. This layered approach allows for quality control at each stage and creates a clear lineage for the final data products.

5.  **Human-in-the-Loop Capability:** While designed for automation, the pipeline also supports human intervention and curation. The `UnifiedEmbryoManager` allows for the addition of detailed, manual annotations such as phenotypes, genotypes, and quality-control flags. This combination of automated processing and manual curation ensures that the final data is both comprehensive and accurate.

## High-Level Overview

The segmentation sandbox is a pipeline designed to process raw image data, perform object detection and segmentation, and generate detailed metadata for downstream analysis. The primary goal is to identify and track embryos in a time-series of images, extracting their masks and other relevant information.

The overall workflow is as follows:

1.  **Data Organization:** Raw stitched images are organized into a standardized directory structure. Videos are created from the image sequences, and initial metadata is generated.
2.  **Object Detection:** The pipeline uses GroundingDINO to detect objects of interest (e.g., "individual embryo") in the images. The output is a set of bounding box annotations.
3.  **Segmentation:** Using the bounding boxes from the detection step as prompts, SAM2 (Segment Anything Model 2) is employed to perform instance segmentation, generating masks for each detected object.
4.  **QC and Analysis:** The segmented masks are analyzed for quality control, and additional metadata is generated.
5.  **Mask Export:** The final masks are exported as labeled images using `simple_mask_exporter.py`.
6.  **Metadata Update:** The final metadata, including all annotations and analysis results, is updated and saved.

The entire pipeline is driven by a series of scripts in the `scripts/pipelines` directory, which are executed in a specific order. The `run_pipeline.sh` script orchestrates the execution of these scripts.

A key feature of the sandbox is its reliance on a centralized and strictly enforced ID parsing convention, managed by `parsing_utils.py`. This ensures that all entities (experiments, videos, images, embryos, snips) are consistently identified and referenced throughout the pipeline.

## Core Utilities and Scripts

### `scripts/data_organization/data_organizer.py`

This script is the first step in the pipeline. It takes raw stitched images from experiment directories and organizes them into a standardized structure. Its main responsibilities are:

*   **Parsing filenames:** It extracts well ID and frame number from the raw image filenames.
*   **Image conversion:** It converts the raw images (e.g., PNG, TIF) to JPEG format.
*   **Video creation:** It creates MP4 videos from the JPEG sequences for each well, with frame numbers overlaid on the video.
*   **Metadata generation:** It scans the organized directory structure and generates an initial `experiment_metadata.json` file, which serves as the input for the rest of the pipeline.

### `scripts/detection_segmentation/grounded_dino_utils.py`

This script is responsible for object detection using the GroundingDINO model. It takes the organized images and the `experiment_metadata.json` file as input.

*   **Inference:** It runs the GroundingDINO model on the images with a given text prompt (e.g., "individual embryo") to detect objects.
*   **Annotation generation:** It generates a JSON file with bounding box annotations for each detected object in each image. This file is used as input for the segmentation step.
*   **High-quality annotation filtering:** It includes functionality to filter the raw detections to produce a set of high-quality annotations based on confidence scores and Intersection over Union (IoU) thresholds.

### `scripts/detection_segmentation/sam2_utils.py`

This script performs instance segmentation using the SAM2 model. It uses the high-quality annotations from the GroundingDINO step as prompts.

*   **Video processing:** It processes each video, using the bounding boxes from a "seed frame" to initialize the tracking of objects.
*   **Segmentation:** It propagates the segmentation masks from the seed frame to the rest of the video, generating a mask for each embryo in each frame.
*   **Output generation:** It produces a detailed JSON file containing the segmentation results, including the masks in Run-Length Encoded (RLE) format, bounding boxes, and other metadata for each segmented embryo.

### `scripts/utils/parsing_utils.py`

This is a critical utility module that provides a centralized and consistent way to parse and build entity IDs. It enforces the strict ID hierarchy and naming conventions described in `docs/id_parsing_conventions.md`. All other scripts in the pipeline use this module for ID manipulation, ensuring data integrity and preventing parsing errors.

### `scripts/metadata/experiment_metadata.py`

This module provides a class for managing the `experiment_metadata.json` file. It allows for adding, updating, and querying metadata for experiments, videos, and images. It also includes functionality to scan the organized directory structure and update the metadata accordingly.

### `scripts/utils/simple_mask_exporter.py`

This script is used by the `06_export_masks.py` pipeline step to export the segmentation masks from the `GroundedSam2Annotations.json` file into a user-friendly image format. Its key features are:

*   **Labeled Image Generation:** It creates labeled images (e.g., PNG, TIFF) where the pixel value of each mask corresponds to the embryo number (e.g., all pixels for embryo `_e01` have a value of 1, `_e02` have a value of 2, etc.).
*   **CRUD Operations:** It maintains a manifest file (`mask_export_manifest.json`) to keep track of which masks have already been exported, avoiding redundant processing.
*   **Flexible Export:** It allows for exporting masks for all experiments or a specified subset, and can overwrite existing files if needed.

### `scripts/pipelines/`

This directory contains the individual scripts that make up the main pipeline, executed in numerical order by `run_pipeline.sh`:

*   `01_prepare_videos.py`: Initializes the data organization process.
*   `03_gdino_detection.py`: Runs the GroundingDINO detection.
*   `04_sam2_segmentation.py`: Runs the SAM2 segmentation.
*   `05_sam2_qc_analysis.py`: Performs quality control on the segmentation results.
*   `06_export_masks.py`: Exports the final masks using `simple_mask_exporter.py`.
*   `07_embryo_metadata_update.py`: Updates the final metadata.

## File Formats

This section provides a detailed description of the key JSON file formats used in the segmentation sandbox.

### Experiment Metadata (`experiment_metadata.json`)

This file is the central hub for all metadata related to the experiments, videos, and images. It is created by `data_organizer.py` and used by all subsequent scripts in the pipeline.

```json
{
    "file_info": {
        "creation_time": "2025-08-25T12:00:00",
        "version": "1.0",
        "created_by": "ExperimentMetadata",
        "last_updated": "2025-08-25T12:30:00"
    },
    "experiments": {
        "20240411": {
            "experiment_id": "20240411",
            "videos": {
                "20240411_A01": {
                    "video_id": "20240411_A01",
                    "well_id": "A01",
                    "mp4_path": "/path/to/video.mp4",
                    "processed_jpg_images_dir": "/path/to/images",
                    "image_ids": [
                        "20240411_A01_t0000",
                        "20240411_A01_t0001"
                    ],
                    "total_frames": 2,
                    "image_size": [1024, 1024]
                }
            },
            "metadata": {},
            "created_time": "2025-08-25T12:00:00"
        }
    },
    "entity_tracking": {
        "experiments": ["20240411"],
        "videos": ["20240411_A01"],
        "images": ["20240411_A01_t0000", "20240411_A01_t0001"],
        "embryos": [],
        "snips": []
    }
}
```

#### Key Fields Explained

*   **`file_info`**: Basic information about the metadata file itself.
*   **`experiments`**: A dictionary of all experiments, keyed by `experiment_id`.
*   **`videos`**: Inside each experiment, a dictionary of videos, keyed by `video_id`.
    *   **`mp4_path`**: The absolute path to the generated MP4 video file.
    *   **`processed_jpg_images_dir`**: The directory containing the JPEG images for the video.
    *   **`image_ids`**: A list of all image IDs in the video, sorted chronologically.
*   **`entity_tracking`**: A summary of all unique entity IDs, used for validation and tracking.

### GroundedDINO Annotations (`gdino_annotations.json`)

This file stores the output of the GroundingDINO detection step. It contains the bounding box annotations for each image.

```json
{
  "file_info": {
    "creation_time": "2025-08-25T12:00:00",
    "last_updated": "2025-08-25T12:30:00"
  },
  "images": {
    "20240411_A01_t0000": {
      "annotations": [
        {
          "annotation_id": "ann_20250825120000123456",
          "prompt": "individual embryo",
          "model_metadata": {
            "model_config_path": "GroundingDINO_SwinT_OGC.py",
            "model_weights_path": "groundingdino_swint_ogc.pth",
            "model_architecture": "GroundedDINO"
          },
          "inference_params": {
            "box_threshold": 0.35,
            "text_threshold": 0.25
          },
          "timestamp": "2025-08-25T12:00:00",
          "num_detections": 1,
          "detections": [
            {
              "box_xyxy": [0.3, 0.1, 0.7, 0.5],
              "confidence": 0.85,
              "phrase": "individual embryo"
            }
          ]
        }
      ]
    }
  },
  "high_quality_annotations": {
    "20240411": {
      "prompt": "individual embryo",
      "confidence_threshold": 0.5,
      "iou_threshold": 0.5,
      "filtered": {
        "20240411_A01_t0000": [
          {
            "box_xyxy": [0.3, 0.1, 0.7, 0.5],
            "confidence": 0.85,
            "phrase": "individual embryo"
          }
        ]
      }
    }
  }
}
```

#### Key Fields Explained

*   **`images`**: A dictionary of all images with annotations, keyed by `image_id`.
*   **`annotations`**: A list of annotations for each image. Each annotation corresponds to a specific prompt and model configuration.
*   **`detections`**: A list of detected objects, each with a bounding box (`box_xyxy`), confidence score, and the detected phrase.
*   **`high_quality_annotations`**: A separate section containing filtered annotations that meet certain quality criteria (confidence and IoU thresholds). This is the data used as input for the SAM2 segmentation step.

### SAM2 Annotations (`GroundedSam2Annotations.json`)

This is the main output of the segmentation pipeline, containing the final segmentation masks and associated metadata.

```json
{
    "script_version": "sam2_utils.py",
    "creation_time": "2025-08-25T12:00:00",
    "last_updated": "2025-08-25T12:30:00",
    "entity_tracking": {
        "experiments": ["20240411"],
        "videos": ["20240411_A01"],
        "images": ["20240411_A01_t0000", "20240411_A01_t0001"],
        "embryos": ["20240411_A01_e01", "20240411_A01_e02"],
        "snips": ["20240411_A01_e01_s0000", "20240411_A01_e01_s0001"]
    },
    "snip_ids": ["20240411_A01_e01_s0000", "20240411_A01_e01_s0001"],
    "segmentation_format": "rle",
    "experiments": {
        "20240411": {
            "experiment_id": "20240411",
            "videos": {
                "20240411_A01": {
                    "video_id": "20240411_A01",
                    "well_id": "A01",
                    "seed_frame_info": {
                        "seed_frame": "20240411_A01_t0000",
                        "num_embryos": 2
                    },
                    "num_embryos": 2,
                    "frames_processed": 2,
                    "sam2_success": true,
                    "image_ids": {
                        "20240411_A01_t0000": {
                            "image_id": "20240411_A01_t0000",
                            "frame_index": 0,
                            "is_seed_frame": true,
                            "embryos": {
                                "20240411_A01_e01": {
                                    "embryo_id": "20240411_A01_e01",
                                    "snip_id": "20240411_A01_e01_s0000",
                                    "segmentation": {
                                        "counts": "RLE_encoded_string",
                                        "size": [1024, 1024]
                                    },
                                    "bbox": [0.1, 0.2, 0.3, 0.4],
                                    "area": 5000.0,
                                    "mask_confidence": 0.95
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
```

#### Key Fields Explained

*   **`entity_tracking`**: A summary of all the unique entity IDs present in the file, categorized by type. This is used for quick validation and cross-referencing.
*   **`snip_ids`**: A flat list of all snip IDs, which are unique identifiers for each embryo in each frame.
*   **`segmentation_format`**: The format used for storing the segmentation masks. "rle" (Run-Length Encoding) is the standard.
*   **`experiments`**: The top-level container for all processed data, organized by experiment ID.
*   **`videos`**: Within each experiment, data is organized by video ID.
*   **`image_ids`**: Within each video, data is organized by image ID. This contains the segmentation results for each frame.
*   **`embryos`**: Within each image, there is a dictionary of segmented embryos, keyed by their embryo ID.
*   **`segmentation`**: This object contains the actual segmentation mask.
    *   **`counts`**: The Run-Length Encoded (RLE) string representing the mask.
    *   **`size`**: The dimensions of the image [height, width].
*   **`bbox`**: The bounding box of the segmented mask, in normalized [x_min, y_min, x_max, y_max] format.
*   **`area`**: The area of the mask in pixels.
*   **`mask_confidence`**: The confidence score of the segmentation mask.

### Embryo Metadata (`embryo_metadata.json`)

This file stores detailed, human-curated annotations for individual embryos, such as phenotypes, genotypes, and flags

```json
{
    "embryos": {
        "20240411_A01_e01": {
            "genotype": {
                "value": "lmx1b",
                "allele": "sa123",
                "zygosity": "heterozygous",
                "author": "M. Colon",
                "timestamp": "2025-08-25T12:00:00"
            },
            "treatments": {
                "DMSO_1": {
                    "value": "DMSO",
                    "author": "M. Colon",
                    "dosage": "0.1%",
                    "timing": "24hpf"
                }
            },
            "flags": {},
            "notes": "This is a sample note.",
            "metadata": {
                "created": "2025-08-25T12:00:00",
                "last_updated": "2025-08-25T12:30:00"
            },
            "snips": {
                "20240411_A01_e01_s0042": {
                    "phenotype": {
                        "value": "EDEMA",
                        "author": "M. Colon",
                        "notes": "Severe edema observed.",
                        "confidence": 0.9
                    },
                    "flags": [
                        {
                            "value": "MOTION_BLUR",
                            "author": "M. Colon",
                            "flag_type": "MOTION_BLUR",
                            "priority": "medium"
                        }
                    ]
                }
            }
        }
    }
}
```

#### Key Fields Explained

*   **`embryos`**: A dictionary of all embryos with annotations, keyed by `embryo_id`.
*   **`genotype`**: The genetic makeup of the embryo.
*   **`treatments`**: A dictionary of treatments applied to the embryo.
*   **`flags`**: A dictionary of flags raised for the embryo.
*   **`notes`**: Free-text notes about the embryo.
*   **`snips`**: A dictionary of annotations for individual snips (frames) of the embryo, keyed by `snip_id`.
    *   **`phenotype`**: The observed phenotype for the snip.
    *   **`flags`**: A list of flags raised for the snip.

### Exported Mask Format

The `simple_mask_exporter.py` script generates labeled mask images from the `GroundedSam2Annotations.json` file. These masks are saved as images (e.g., PNG, TIFF) in the specified output directory, organized by experiment.

**File Naming Convention:**

The exported mask files follow a specific naming convention that encodes important information directly in the filename:

`{image_id}_masks_emnum_{embryo_count}.{format}`

*   **`image_id`**: The full ID of the source image (e.g., `20240411_A01_t0042`).
*   **`emnum_{embryo_count}`**: The total number of embryos detected in that image.
*   **`format`**: The image format (e.g., `png`).

**Example:**

`20240411_A01_t0042_masks_emnum_2.png`

**Mask Pixel Values:**

Each mask is a labeled image where the pixel values correspond to the embryo number (`embryo_num` from the `embryo_id`). For example:

*   All pixels belonging to the mask of embryo `..._e01` will have a value of **1**.
*   All pixels belonging to the mask of embryo `..._e02` will have a value of **2**.
*   And so on.

This format allows for easy import and analysis of the segmentation masks in standard image processing software.

### Mask Export Manifest (`mask_export_manifest.json`)

This file is created and managed by the `SimpleMaskExporter` to keep track of which masks have been exported. This prevents redundant processing when the export script is run multiple times.

```json
{
    "entity_tracking": {
        "experiments": ["20240411"],
        "videos": ["20240411_A01"],
        "images": ["20240411_A01_t0000"],
        "embryos": ["20240411_A01_e01"],
        "snips": ["20240411_A01_e01_s0000"]
    },
    "exports": {
        "20240411_A01_t0000": {
            "image_id": "20240411_A01_t0000",
            "output_path": "/path/to/output/20240411/masks/20240411_A01_t0000_masks_emnum_1.png",
            "embryo_count": 1,
            "export_timestamp": "2025-08-25T12:30:00",
            "source_file": "/path/to/GroundedSam2Annotations.json",
            "image_shape": [1024, 1024]
        }
    },
    "last_updated": "2025-08-25T12:30:00",
    "total_exported": 1
}
```

#### Key Fields Explained

*   **`entity_tracking`**: A summary of all the entities for which masks have been exported.
*   **`exports`**: A dictionary where each key is an `image_id` and the value contains information about the exported mask file, including its path, the number of embryos, and the timestamp of the export.
*   **`total_exported`**: The total number of images for which masks have been exported.




{
  "file_info": { ... },
  "experiments": {
    "<experiment_id>": {
      "experiment_id": "<experiment_id>",
      "created_time": "ISO-8601 or null",
      "metadata": { ... },               // optional experiment-level info
      "videos": {
        "<video_id>": {
          "video_id": "<video_id>",
          "well_id": "A01",
          "mp4_path": "...",             // from old doc (if present)
          "processed_jpg_images_dir": "...",
          "total_frames": 123,
          "image_size": [H, W],          // from old doc (if present)
          "source_well_metadata_csv": "...",           // from new doc (well-level fields)
          "medium": "E3",
          "genotype": "wildtype",
          "start_age_hpf": 24,
          "embryos_per_well": 1,
        ... and any other info from oriignal gsam piplein a
          "image_ids": {
            "<image_id>": {
              "frame_index": 0,
              "raw_image_data_info": {
                "raw_height_um": 7080.86,
                "raw_width_um": 7080.86,
                "raw_height_px": 2189,
                "raw_width_px": 2189,
                "microscope": "YX1",
                "objective": "Plan Apo λ 4x",
                "bf_channel": 0,
                "nd2_series_num": 1,
                "raw_time_s": 0.977,
                "relative_time_s": 0.0,
                "stitched_image_path": "/path/to/stitched.jpg"
              }
            }
          }
        }
      }
    }
  },
  "entity_tracking": {
    "experiments": [],
    "videos": [],
    "images": [],
    "embryos": [],
    "snips": []