"""
Video generator for MorphSeq pipeline - foundation video with overlays.

Supports progressive enhancement:
1. Foundation video (Module 0): Basic image_id overlay
2. Enhanced videos (later modules): Foundation + additional overlays  
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from .video_config import VideoConfig, COLORBLIND_PALETTE, OVERLAY_COLORS
from .overlay_manager import OverlayManager

class VideoGenerator:
    """
    Creates foundation videos and enhanced videos with overlays.
    
    Philosophy: Create foundation video once, then overlay annotations for speed.
    """
    
    def __init__(self, config: Optional[VideoConfig] = None):
        self.config = config or VideoConfig()
        self.overlay_manager = OverlayManager(self.config)
        
    def create_foundation_video(self, 
                              jpeg_paths: List[Path], 
                              video_path: Path,
                              video_id: str,
                              verbose: bool = True) -> bool:
        """
        Create basic video with just image_id overlay (10% down from top-right).
        This is the foundation that other modules will enhance.
        """
        if not jpeg_paths:
            if verbose:
                print("❌ No JPEG files to create video from")
            return False
            
        # Get video dimensions from first frame
        first_frame = cv2.imread(str(jpeg_paths[0]))
        if first_frame is None:
            if verbose:
                print("❌ Could not read first frame for video creation")
            return False
            
        height, width = first_frame.shape[:2]
        
        if verbose:
            print(f"🎬 Creating foundation video {width}x{height} with {len(jpeg_paths)} frames...")
            
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.config.CODEC)
        video_writer = cv2.VideoWriter(str(video_path), fourcc, self.config.FPS, (width, height))
        
        if not video_writer.isOpened():
            if verbose:
                print(f"❌ Could not open video writer for {video_path}")
            return False
            
        frames_written = 0
        
        for jpeg_path in sorted(jpeg_paths):
            frame = cv2.imread(str(jpeg_path))
            if frame is None:
                continue
                
            # Add image_id overlay (10% down from top-right)
            frame_num = jpeg_path.stem
            image_id = f"{video_id}_t{frame_num}"
            
            frame_with_overlay = self._add_image_id_overlay(frame, image_id)
            
            video_writer.write(frame_with_overlay)
            frames_written += 1
            
        video_writer.release()
        
        if verbose:
            print(f"✅ Foundation video created: {video_path.name} ({frames_written} frames)")
            
        return True
        
    def create_enhanced_video(self,
                            foundation_video_path: Path,
                            output_video_path: Path, 
                            overlay_dict: Dict[str, Any],
                            overlay_type: str = "detection",
                            verbose: bool = True) -> bool:
        """
        Create enhanced video by adding overlays to foundation video.
        
        Args:
            foundation_video_path: Path to existing foundation video
            output_video_path: Where to save enhanced video
            overlay_dict: {image_id: overlay_data} mapping
            overlay_type: 'detection', 'mask', 'metadata', 'qc_flags'
        """
        if not foundation_video_path.exists():
            if verbose:
                print(f"❌ Foundation video not found: {foundation_video_path}")
            return False
            
        # Open foundation video for reading
        cap = cv2.VideoCapture(str(foundation_video_path))
        if not cap.isOpened():
            if verbose:
                print(f"❌ Could not open foundation video: {foundation_video_path}")
            return False
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize output video writer
        fourcc = cv2.VideoWriter_fourcc(*self.config.CODEC)
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            if verbose:
                print(f"❌ Could not open output video writer: {output_video_path}")
            cap.release()
            return False
            
        if verbose:
            print(f"🎨 Creating enhanced video with {overlay_type} overlays...")
            
        frame_idx = 0
        frames_processed = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Generate image_id for this frame (assuming standard naming)
            frame_num = str(frame_idx).zfill(4)
            # Extract video_id from foundation video name (remove .mp4)
            video_id = foundation_video_path.stem
            image_id = f"{video_id}_t{frame_num}"
            
            # Add overlays if available for this image_id
            if image_id in overlay_dict:
                frame = self.overlay_manager.add_overlay(
                    frame, 
                    overlay_dict[image_id], 
                    overlay_type
                )
                
            out.write(frame)
            frames_processed += 1
            frame_idx += 1
            
        cap.release()
        out.release()
        
        if verbose:
            overlays_applied = sum(1 for image_id in overlay_dict.keys() 
                                 if image_id.startswith(video_id))
            print(f"✅ Enhanced video created: {output_video_path.name}")
            print(f"   📊 {frames_processed} frames processed, {overlays_applied} frames with overlays")
            
        return True
        
    def _add_image_id_overlay(self, frame: np.ndarray, image_id: str) -> np.ndarray:
        """
        Add image_id text overlay at top-right, 10% down from top.
        Matches original implementation positioning.
        """
        height, width = frame.shape[:2]
        
        # Calculate text size
        (text_width, text_height), _ = cv2.getTextSize(
            image_id, 
            self.config.FONT, 
            self.config.FONT_SCALE, 
            self.config.FONT_THICKNESS
        )
        
        # Position: 10% down from top, right-aligned with margin
        text_x = width - text_width - self.config.IMAGE_ID_MARGIN_RIGHT
        text_y = int(height * self.config.IMAGE_ID_Y_PERCENTAGE)
        
        # Add semi-transparent background for better readability
        overlay = frame.copy()
        cv2.rectangle(
            overlay, 
            (text_x - 5, text_y - text_height - 5), 
            (text_x + text_width + 5, text_y + 5), 
            self.config.TEXT_BACKGROUND_COLOR, 
            -1
        )
        
        # Blend background
        frame = cv2.addWeighted(
            overlay, 
            self.config.TEXT_BACKGROUND_ALPHA, 
            frame, 
            1 - self.config.TEXT_BACKGROUND_ALPHA, 
            0
        )
        
        # Add text
        cv2.putText(
            frame, 
            image_id, 
            (text_x, text_y), 
            self.config.FONT, 
            self.config.FONT_SCALE, 
            self.config.TEXT_COLOR, 
            self.config.FONT_THICKNESS
        )
        
        return frame
        
    def create_sam2_eval_video_from_results(self,
                                           results_json_path: Path,
                                           experiment_id: str,
                                           video_id: str,
                                           output_video_path: Path,
                                           show_bbox: bool = False,
                                           show_mask: bool = True,
                                           show_metrics: bool = True,
                                           show_qc_flags: bool = False,
                                           verbose: bool = True) -> bool:
        """
        Create SAM2 evaluation video from results JSON.
        
        Args:
            results_json_path: Path to GroundedSam2Annotations.json
            experiment_id: Experiment ID (e.g., "20250612_30hpf_ctrl_atf6")
            video_id: Video ID (e.g., "20250612_30hpf_ctrl_atf6_A01")
            output_video_path: Where to save the MP4 file
            show_bbox: Show bounding boxes (default: False)
            show_mask: Show segmentation masks (default: True)
            show_metrics: Show QC metrics (default: True)
            show_qc_flags: Show quality control flags (default: False)
            verbose: Print progress
            
        Returns:
            True if successful, False otherwise
        """
        if verbose:
            print(f"🎬 Creating SAM2 evaluation video for {video_id}")
            
        # Load SAM2 results
        try:
            with open(results_json_path, 'r') as f:
                sam2_data = json.load(f)
        except Exception as e:
            if verbose:
                print(f"❌ Failed to load SAM2 results: {e}")
            return False
            
        # Find experiment and video data
        experiments = sam2_data.get("experiments", {})
        if experiment_id not in experiments:
            if verbose:
                print(f"❌ Experiment {experiment_id} not found in results")
            return False
            
        exp_data = experiments[experiment_id]
        videos = exp_data.get("videos", {})
        # print(videos)
        if video_id not in videos:
            if verbose:
                print(f"❌ Video {video_id} not found in experiment")
            return False
            
        video_data = videos[video_id]
        images = video_data.get("image_ids", {})
        print(video_data)
  

        # Find video directory - use experiment metadata to locate images
        # Standard path structure: data/raw_data_organized/{experiment_id}/images/{video_id}/
        video_dir = Path("data/raw_data_organized") / experiment_id / "images" / video_id
        
        if not video_dir.exists():
            if verbose:
                print(f"❌ Video directory not found: {video_dir}")
            return False
            
        # Get sorted image files
        image_files = sorted(list(video_dir.glob("*.jpg")))
        if not image_files:
            if verbose:
                print(f"❌ No JPEG files found in {video_dir}")
            return False
            
        if verbose:
            print(f"📁 Found {len(image_files)} frames in {video_dir}")
            # print(f"📊 Processing {len(images)} frames with SAM2 data")
            
        # Get video dimensions from first frame
        first_frame = cv2.imread(str(image_files[0]))
        if first_frame is None:
            if verbose:
                print("❌ Could not read first frame")
            return False
            
        height, width = first_frame.shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.config.CODEC)
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, self.config.FPS, (width, height))
        
        if not video_writer.isOpened():
            if verbose:
                print(f"❌ Could not open video writer for {output_video_path}")
            return False
            
        frames_written = 0
        frames_with_overlays = 0
        
        # Process each frame
        for image_file in image_files:
            frame = cv2.imread(str(image_file))
            if frame is None:
                continue
                
            # Generate image_id from filename - use the actual filename as it appears in SAM2 data
            frame_num = image_file.stem  # This is already the full frame name like "20250612_30hpf_ctrl_atf6_A01_ch00_t0000"
            image_id = frame_num  # Don't add video_id prefix since it's already in the filename
            
            # Add image_id overlay (foundation)
            frame = self._add_image_id_overlay(frame, image_id)
            
            # Add SAM2 overlays if data exists for this image
            if image_id in images:
                image_data = images[image_id]
                embryos = image_data.get("embryos", {})
                
                if embryos:
                    frame = self.overlay_manager.add_sam2_embryos_overlay(
                        frame,
                        embryos,
                        show_bbox=show_bbox,
                        show_mask=show_mask,
                        show_metrics=show_metrics
                    )
                    frames_with_overlays += 1
                elif verbose:
                    print(f"⚠️ No embryos found for image {image_id}")
            elif verbose:
                print(f"⚠️ Image {image_id} not found in SAM2 data. Available: {list(images.keys())[:3]}...")
                    
                # Add QC flags overlay if requested and available
                # Add QC flags overlay if requested and available
                if show_qc_flags and image_data.get("qc_flags"):
                    frame = self.overlay_manager.add_overlay(
                        frame,
                        image_data["qc_flags"],
                        "qc_flags"
                    )
                    
            video_writer.write(frame)
            frames_written += 1
            
        video_writer.release()
        
        if verbose:
            print(f"✅ SAM2 evaluation video created: {output_video_path.name}")
            print(f"   📊 {frames_written} frames written, {frames_with_overlays} with SAM2 overlays")
            
        return True
        
    @staticmethod
    def extract_video_id_from_path(video_path: Path) -> str:
        """Extract video_id from video file path."""
        return video_path.stem
        
    @staticmethod
    def generate_image_id(video_id: str, frame_number: int) -> str:
        """Generate standard image_id from video_id and frame number."""
        return f"{video_id}_t{str(frame_number).zfill(4)}"
