#!/usr/bin/env python3
"""
Script to convert normalized center-based box_xywh to absolute corner-based box_xyxy
in GDINO annotation JSON files. It backs up original files with .bak extension.
"""
import os
import json
import shutil
import argparse
from PIL import Image
import sys
import os  # ensure os available for sys.path manipulation
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts", "utils"))
import experiment_metadata_utils as emu

def load_image_sizes(data, img_root=None):
    """
    Build a mapping from image_id to (width, height). If width/height not in JSON,
    load the image from img_root/file_name to get dimensions.
    """
    id2size = {}
    images = data.get('images') or data.get('imgs') or []
    for img in images:
        img_id = img.get('id')
        width = img.get('width')
        height = img.get('height')
        file_name = img.get('file_name') or img.get('file')
        if width and height:
            id2size[img_id] = (width, height)
        elif file_name:
            img_path = os.path.join(img_root, file_name) if img_root else file_name
            with Image.open(img_path) as im:
                id2size[img_id] = (im.width, im.height)
    return id2size

def build_size_map_from_metadata(data, metadata, img_root=None):
    """
    Build a mapping from image_id to (width, height) via experiment metadata or by loading the JPEGs.
    """
    id2size = {}
    # images keys correspond to image_ids
    for image_id in data.get('images', {}):
        # attempt resolution from metadata
        try:
            comps = emu.parse_image_id(image_id)
            vid = comps['video_id']
            info = emu.get_video_info(vid, metadata)
            if info and 'video_resolution' in info:
                id2size[image_id] = tuple(info['video_resolution'])
                continue
        except Exception:
            pass
        # fallback: load actual JPEG
        try:
            img_path = emu.get_image_id_paths(image_id, metadata)
            with Image.open(img_path) as im:
                id2size[image_id] = (im.width, im.height)
        except Exception:
            pass
    return id2size

def convert_box_xywh_to_xyxy(box, width, height):
    """
    Convert normalized center-based [x_c, y_c, w, h]
    to absolute corner-based [x0, y0, x1, y1].
    """
    x_c, y_c, w_norm, h_norm = box
    x0 = (x_c - w_norm/2) * width
    y0 = (y_c - h_norm/2) * height
    x1 = (x_c + w_norm/2) * width
    y1 = (y_c + h_norm/2) * height
    return [x0, y0, x1, y1]

def recurse_and_convert(obj, size_map):
    """
    Recursively search for dicts with box_xywh and image_id,
    replace with box_xyxy using size_map.
    """
    if isinstance(obj, dict):
        if 'image_id' in obj and 'box_xywh' in obj:
            img_id = obj['image_id']
            box = obj['box_xywh']
            if img_id in size_map:
                w, h = size_map[img_id]
                obj['box_xyxy'] = convert_box_xywh_to_xyxy(box, w, h)
                del obj['box_xywh']
        for v in obj.values():
            recurse_and_convert(v, size_map)
    elif isinstance(obj, list):
        for item in obj:
            recurse_and_convert(item, size_map)

def convert_metadata_annotations(data, size_map):
    # convert detections under data['images']
    for image_id, img_entry in data.get('images', {}).items():
        size = size_map.get(image_id)
        if not size:
            continue
        for ann in img_entry.get('annotations', []):
            for det in ann.get('detections', []):
                det['box_xyxy'] = convert_box_xywh_to_xyxy(det['box_xywh'], *size)
                del det['box_xywh']
    # convert under high_quality_annotations filtered
    for exp in data.get('high_quality_annotations', {}).values():
        for image_id, det_list in exp.get('filtered', {}).items():
            size = size_map.get(image_id)
            if not size:
                continue
            for det in det_list:
                det['box_xyxy'] = convert_box_xywh_to_xyxy(det['box_xywh'], *size)
                del det['box_xywh']
    return data

def process_file(path, metadata_path=None, img_root=None):
    """
    Backup, convert boxes in file, and overwrite.
    """
    backup_path = path + '.bak'
    shutil.copy2(path, backup_path)
    print(f"→ Backed up {path} to {backup_path}")

    with open(path, 'r') as f:
        data = json.load(f)

    if metadata_path:
        metadata = emu.load_experiment_metadata(metadata_path)
        size_map = build_size_map_from_metadata(data, metadata, img_root)
        data = convert_metadata_annotations(data, size_map)
    else:
        size_map = load_image_sizes(data, img_root)
        recurse_and_convert(data, size_map)

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✔ Converted box_xywh to box_xyxy in {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert normalized box_xywh -> box_xyxy in GDINO JSON files"
    )
    parser.add_argument(
        'files', nargs='+', help='Path(s) to JSON file(s) to convert'
    )
    parser.add_argument(
        '--metadata', '-m', help='Path to experiment_metadata.json', default=None
    )
    # no --images option: root dirs come from metadata via get_image_id_paths
    args = parser.parse_args()

    for fp in args.files:
        if os.path.isfile(fp):
            process_file(fp, args.metadata)
        else:
            print(f"✖ File not found: {fp}")

if __name__ == '__main__':
    main()


