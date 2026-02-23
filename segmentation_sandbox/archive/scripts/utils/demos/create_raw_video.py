#!/usr/bin/env python3
"""
Create a video from raw frames in a directory.
Usage:
    python create_raw_video.py /path/to/raw_frames_dir /path/to/output_video.mp4 [--fps FPS]
"""
import sys
from pathlib import Path
import cv2
import glob


def create_video_from_image_dir(images_dir, output_video, fps=8):
    img_paths = sorted(glob.glob(str(Path(images_dir) / '*.jpg')))
    if not img_paths:
        raise RuntimeError(f"No .jpg files found in {images_dir}")
    frame = cv2.imread(img_paths[0])
    if frame is None:
        raise RuntimeError(f"Cannot read {img_paths[0]}")
    h, w = frame.shape[:2]
    if h % 2 or w % 2:
        h, w = h - (h % 2), w - (w % 2)
        frame = frame[:h, :w]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Failed to open video writer")
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: skipping unreadable {img_path}")
            continue
        img = img[:h, :w]
        writer.write(img)
    writer.release()
    print(f"✅ Video saved to {output_video}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create video from raw frames directory")
    parser.add_argument("raw_frames_dir", type=str, help="Directory containing raw .jpg frames")
    parser.add_argument("output_video", type=str, help="Output video file path")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second")
    args = parser.parse_args()
    create_video_from_image_dir(args.raw_frames_dir, args.output_video, args.fps)
