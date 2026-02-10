"""
Complete Embryo Mask Export and QC Pipeline Script
Integrates mask export, QC, metadata, and manifest generation for end-to-end workflow.
"""

import argparse
from pathlib import Path
from scripts.utils.mask_export_utils import EmbryoMaskExporter
from scripts.gsam_qc_class import GSAMQualityControl
from scripts.utils.embryo_metada_dev_instruction.embryo_metadata_refactored import EmbryoMetadata


def main():
    parser = argparse.ArgumentParser(description="Embryo Mask Export and QC Pipeline")
    parser.add_argument("--sam_path", required=True, help="Path to SAM2 annotation JSON")
    parser.add_argument("--embryo_metadata_path", required=True, help="Path to embryo metadata JSON")
    parser.add_argument("--output_dir", required=True, help="Directory to export masks and manifest")
    parser.add_argument("--author", default="QC_Pipeline", help="QC author name")
    parser.add_argument("--qc_only", action="store_true", help="Run QC only (skip mask export)")
    args = parser.parse_args()

    sam_path = Path(args.sam_path)
    embryo_metadata_path = Path(args.embryo_metadata_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mask export
    if not args.qc_only:
        exporter = EmbryoMaskExporter(sam_path, output_dir, verbose=True)
        manifest = exporter.export_all_masks()
        manifest_path = output_dir / "mask_export_manifest.json"
        exporter.save_manifest(manifest_path)
        print(f"Mask export complete. Manifest saved to {manifest_path}")

    # QC
    qc = GSAMQualityControl(str(sam_path), str(embryo_metadata_path), verbose=True)
    qc.check_segmentation_variability(args.author)
    qc.check_mask_on_edge(args.author)
    qc.check_detection_failure(args.author)
    qc.push_flags_to_metadata()
    qc.save_flags(output_dir / "gsam_qc_flags.json")
    print(f"QC complete. Flags saved to {output_dir / 'gsam_qc_flags.json'}")

    # Save updated metadata
    qc.embryo_metadata.save(str(output_dir / "embryo_metadata_with_qc.json"))
    print(f"Updated metadata saved to {output_dir / 'embryo_metadata_with_qc.json'}")

if __name__ == "__main__":
    main()
