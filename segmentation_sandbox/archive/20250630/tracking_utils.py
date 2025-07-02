#!/usr/bin/env python
"""
Tracking utilities for the MorphSeq embryo segmentation pipeline.
Handles embryo tracking across frames, death detection, and trajectory analysis.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd
from scipy.spatial.distance import cdist
from .detection_utils import Detection
from .mask_utils import calculate_mask_overlap, get_mask_properties


@dataclass
class EmbryoTrack:
    """Data class for embryo tracking information."""
    track_id: int
    detections: Dict[int, Detection] = field(default_factory=dict)  # frame_idx -> Detection
    first_frame: int = -1
    last_frame: int = -1
    is_alive: bool = True
    death_frame: Optional[int] = None
    confidence_history: List[float] = field(default_factory=list)
    position_history: List[Tuple[float, float]] = field(default_factory=list)  # (x, y) centroids
    
    def add_detection(self, frame_idx: int, detection: Detection):
        """Add detection to track."""
        self.detections[frame_idx] = detection
        
        if self.first_frame == -1 or frame_idx < self.first_frame:
            self.first_frame = frame_idx
        if frame_idx > self.last_frame:
            self.last_frame = frame_idx
        
        self.confidence_history.append(detection.confidence)
        
        # Calculate centroid from bbox
        bbox = detection.bbox
        centroid_x = (bbox[0] + bbox[2]) / 2
        centroid_y = (bbox[1] + bbox[3]) / 2
        self.position_history.append((centroid_x, centroid_y))
    
    def get_detection(self, frame_idx: int) -> Optional[Detection]:
        """Get detection for specific frame."""
        return self.detections.get(frame_idx)
    
    def get_last_position(self) -> Optional[Tuple[float, float]]:
        """Get last known position."""
        if self.position_history:
            return self.position_history[-1]
        return None
    
    def get_frame_gap(self, current_frame: int) -> int:
        """Get number of frames since last detection."""
        if not self.detections:
            return current_frame
        
        last_detection_frame = max(self.detections.keys())
        return current_frame - last_detection_frame
    
    def mark_dead(self, death_frame: int):
        """Mark embryo as dead."""
        self.is_alive = False
        self.death_frame = death_frame
    
    def get_trajectory_data(self) -> Dict[str, Any]:
        """Get trajectory data for analysis."""
        if not self.detections:
            return {}
        
        frames = sorted(self.detections.keys())
        
        trajectory = {
            'track_id': self.track_id,
            'first_frame': self.first_frame,
            'last_frame': self.last_frame,
            'total_frames': len(frames),
            'is_alive': self.is_alive,
            'death_frame': self.death_frame,
            'frames': frames,
            'positions': [self.position_history[i] for i in range(len(frames))],
            'confidences': [self.confidence_history[i] for i in range(len(frames))],
            'mean_confidence': np.mean(self.confidence_history) if self.confidence_history else 0.0,
            'std_confidence': np.std(self.confidence_history) if self.confidence_history else 0.0
        }
        
        return trajectory


class EmbryoTracker:
    """
    Handles embryo tracking across video frames.
    Uses position and appearance similarity for tracking.
    """
    
    def __init__(self, config_params: Dict[str, Any]):
        """
        Initialize embryo tracker.
        
        Args:
            config_params: Tracking configuration parameters
        """
        self.config = config_params
        self.max_distance = config_params.get('max_tracking_distance', 0.1)  # Normalized distance
        self.num_consecutive_undetected = config_params.get('num_consecutive_undetected', 4)
        self.max_tracking_gap = config_params.get('max_tracking_gap', 3)
        
        self.tracks: Dict[int, EmbryoTrack] = {}
        self.next_track_id = 1
        self.current_frame = 0
        
    def update(self, frame_idx: int, detections: List[Detection]) -> List[EmbryoTrack]:
        """
        Update tracks with new detections.
        
        Args:
            frame_idx: Current frame index
            detections: List of detections for current frame
            
        Returns:
            List of active tracks
        """
        self.current_frame = frame_idx
        
        if not detections:
            # No detections, check for dead embryos
            self._check_for_dead_embryos(frame_idx)
            return list(self.tracks.values())
        
        # Get active tracks (not dead and recently detected)
        active_tracks = [track for track in self.tracks.values() 
                        if track.is_alive and track.get_frame_gap(frame_idx) <= self.max_tracking_gap]
        
        if not active_tracks:
            # No active tracks, create new tracks for all detections
            for detection in detections:
                self._create_new_track(frame_idx, detection)
        else:
            # Match detections to existing tracks
            matched_pairs, unmatched_detections, unmatched_tracks = self._match_detections_to_tracks(
                detections, active_tracks, frame_idx
            )
            
            # Update matched tracks
            for detection_idx, track in matched_pairs:
                track.add_detection(frame_idx, detections[detection_idx])
            
            # Create new tracks for unmatched detections
            for detection_idx in unmatched_detections:
                self._create_new_track(frame_idx, detections[detection_idx])
        
        # Check for dead embryos
        self._check_for_dead_embryos(frame_idx)
        
        return list(self.tracks.values())
    
    def _create_new_track(self, frame_idx: int, detection: Detection) -> EmbryoTrack:
        """Create new track for detection."""
        track = EmbryoTrack(track_id=self.next_track_id)
        track.add_detection(frame_idx, detection)
        
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1
        
        return track
    
    def _match_detections_to_tracks(self, detections: List[Detection], 
                                  tracks: List[EmbryoTrack], 
                                  frame_idx: int) -> Tuple[List[Tuple[int, EmbryoTrack]], List[int], List[EmbryoTrack]]:
        """
        Match detections to existing tracks using Hungarian algorithm approximation.
        
        Args:
            detections: Current detections
            tracks: Active tracks
            frame_idx: Current frame index
            
        Returns:
            Tuple of (matched_pairs, unmatched_detection_indices, unmatched_tracks)
        """
        if not detections or not tracks:
            return [], list(range(len(detections))), tracks
        
        # Calculate cost matrix (distances between detections and track predictions)
        cost_matrix = np.zeros((len(detections), len(tracks)))
        
        for i, detection in enumerate(detections):
            det_pos = self._get_detection_position(detection)
            
            for j, track in enumerate(tracks):
                track_pos = track.get_last_position()
                if track_pos is None:
                    cost_matrix[i, j] = float('inf')
                else:
                    # Calculate distance
                    distance = np.sqrt((det_pos[0] - track_pos[0])**2 + (det_pos[1] - track_pos[1])**2)
                    
                    # Add mask similarity if available
                    similarity_bonus = 0.0
                    if detection.mask is not None:
                        last_detection = track.get_detection(track.last_frame)
                        if last_detection and last_detection.mask is not None:
                            mask_overlap = calculate_mask_overlap(detection.mask, last_detection.mask)
                            similarity_bonus = mask_overlap * 0.05  # Small bonus for mask similarity
                    
                    cost_matrix[i, j] = distance - similarity_bonus
        
        # Simple greedy matching (could be improved with Hungarian algorithm)
        matched_pairs = []
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(tracks)
        
        # Sort by cost and greedily match
        costs = []
        for i in range(len(detections)):
            for j, track in enumerate(tracks):
                if cost_matrix[i, j] <= self.max_distance:
                    costs.append((cost_matrix[i, j], i, j, track))
        
        costs.sort()  # Sort by cost (lowest first)
        
        for cost, det_idx, track_idx, track in costs:
            if det_idx in unmatched_detections and track in unmatched_tracks:
                matched_pairs.append((det_idx, track))
                unmatched_detections.remove(det_idx)
                unmatched_tracks.remove(track)
        
        return matched_pairs, list(unmatched_detections), list(unmatched_tracks)
    
    def _get_detection_position(self, detection: Detection) -> Tuple[float, float]:
        """Get position (centroid) of detection."""
        bbox = detection.bbox
        centroid_x = (bbox[0] + bbox[2]) / 2
        centroid_y = (bbox[1] + bbox[3]) / 2
        return (centroid_x, centroid_y)
    
    def _check_for_dead_embryos(self, frame_idx: int):
        """Check for embryos that should be marked as dead."""
        for track in self.tracks.values():
            if track.is_alive:
                gap = track.get_frame_gap(frame_idx)
                if gap >= self.num_consecutive_undetected:
                    track.mark_dead(frame_idx - gap + 1)  # Death frame is when it was last seen + 1
    
    def get_active_tracks(self, frame_idx: int) -> List[EmbryoTrack]:
        """Get tracks that are active (alive and recently detected)."""
        return [track for track in self.tracks.values() 
                if track.is_alive and track.get_frame_gap(frame_idx) <= self.max_tracking_gap]
    
    def get_all_tracks(self) -> List[EmbryoTrack]:
        """Get all tracks."""
        return list(self.tracks.values())
    
    def get_tracks_summary(self) -> Dict[str, Any]:
        """Get summary of tracking results."""
        all_tracks = self.get_all_tracks()
        
        summary = {
            'total_tracks': len(all_tracks),
            'alive_tracks': len([t for t in all_tracks if t.is_alive]),
            'dead_tracks': len([t for t in all_tracks if not t.is_alive]),
            'mean_track_length': 0.0,
            'std_track_length': 0.0,
            'longest_track': 0,
            'shortest_track': 0
        }
        
        if all_tracks:
            track_lengths = [len(track.detections) for track in all_tracks]
            summary['mean_track_length'] = np.mean(track_lengths)
            summary['std_track_length'] = np.std(track_lengths)
            summary['longest_track'] = max(track_lengths)
            summary['shortest_track'] = min(track_lengths)
        
        return summary


def analyze_trajectories(tracks: List[EmbryoTrack]) -> pd.DataFrame:
    """
    Analyze embryo trajectories and create summary DataFrame.
    
    Args:
        tracks: List of embryo tracks
        
    Returns:
        DataFrame with trajectory analysis
    """
    trajectory_data = []
    
    for track in tracks:
        if not track.detections:
            continue
        
        # Basic trajectory info
        traj_info = track.get_trajectory_data()
        
        # Calculate additional metrics
        positions = traj_info['positions']
        if len(positions) > 1:
            # Calculate total distance traveled
            distances = []
            for i in range(1, len(positions)):
                dist = np.sqrt((positions[i][0] - positions[i-1][0])**2 + 
                             (positions[i][1] - positions[i-1][1])**2)
                distances.append(dist)
            
            traj_info['total_distance'] = sum(distances)
            traj_info['mean_velocity'] = np.mean(distances) if distances else 0.0
            traj_info['max_velocity'] = max(distances) if distances else 0.0
        else:
            traj_info['total_distance'] = 0.0
            traj_info['mean_velocity'] = 0.0
            traj_info['max_velocity'] = 0.0
        
        # Calculate movement variability
        if len(positions) > 2:
            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            traj_info['x_variance'] = np.var(x_coords)
            traj_info['y_variance'] = np.var(y_coords)
        else:
            traj_info['x_variance'] = 0.0
            traj_info['y_variance'] = 0.0
        
        trajectory_data.append(traj_info)
    
    if trajectory_data:
        df = pd.DataFrame(trajectory_data)
    else:
        # Empty DataFrame with expected columns
        df = pd.DataFrame(columns=[
            'track_id', 'first_frame', 'last_frame', 'total_frames', 'is_alive', 'death_frame',
            'mean_confidence', 'std_confidence', 'total_distance', 'mean_velocity', 'max_velocity',
            'x_variance', 'y_variance'
        ])
    
    return df


def detect_tracking_anomalies(tracks: List[EmbryoTrack], 
                            config_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Detect anomalies in tracking results.
    
    Args:
        tracks: List of embryo tracks
        config_params: Configuration parameters with thresholds
        
    Returns:
        List of anomaly reports
    """
    anomalies = []
    
    # Thresholds from config
    min_track_length = config_params.get('min_track_length', 5)
    max_velocity_threshold = config_params.get('max_velocity_threshold', 0.1)
    high_variance_threshold = config_params.get('high_variance_threshold', 0.05)
    
    for track in tracks:
        track_anomalies = []
        
        # Check track length
        if len(track.detections) < min_track_length:
            track_anomalies.append({
                'type': 'SHORT_TRACK',
                'message': f'Track length {len(track.detections)} below threshold {min_track_length}'
            })
        
        # Check for sudden movements (high velocity)
        positions = track.position_history
        if len(positions) > 1:
            for i in range(1, len(positions)):
                velocity = np.sqrt((positions[i][0] - positions[i-1][0])**2 + 
                                 (positions[i][1] - positions[i-1][1])**2)
                if velocity > max_velocity_threshold:
                    track_anomalies.append({
                        'type': 'HIGH_VELOCITY',
                        'message': f'High velocity {velocity:.4f} at frame transition',
                        'frame': track.first_frame + i
                    })
        
        # Check for high position variance (erratic movement)
        if len(positions) > 2:
            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            x_var = np.var(x_coords)
            y_var = np.var(y_coords)
            
            if x_var > high_variance_threshold or y_var > high_variance_threshold:
                track_anomalies.append({
                    'type': 'HIGH_POSITION_VARIANCE',
                    'message': f'High position variance: x={x_var:.4f}, y={y_var:.4f}'
                })
        
        # Check confidence variability
        if len(track.confidence_history) > 1:
            conf_std = np.std(track.confidence_history)
            conf_mean = np.mean(track.confidence_history)
            conf_cv = conf_std / conf_mean if conf_mean > 0 else 0
            
            if conf_cv > high_variance_threshold:
                track_anomalies.append({
                    'type': 'HIGH_CONFIDENCE_VARIANCE',
                    'message': f'High confidence variability: CV={conf_cv:.4f}'
                })
        
        if track_anomalies:
            anomalies.append({
                'track_id': track.track_id,
                'anomalies': track_anomalies
            })
    
    return anomalies


# Example usage and testing
if __name__ == "__main__":
    # Test tracking utilities
    print("Testing tracking utilities...")
    
    # Create sample detections
    detections_frame1 = [
        Detection([0.1, 0.1, 0.2, 0.2], 0.8),
        Detection([0.5, 0.5, 0.6, 0.6], 0.9),
    ]
    
    detections_frame2 = [
        Detection([0.12, 0.12, 0.22, 0.22], 0.7),  # Same embryo, slightly moved
        Detection([0.52, 0.48, 0.62, 0.58], 0.85),  # Same embryo, slightly moved
    ]
    
    # Initialize tracker
    config = {
        'max_tracking_distance': 0.1,
        'num_consecutive_undetected': 3
    }
    
    tracker = EmbryoTracker(config)
    
    # Update with detections
    tracks1 = tracker.update(0, detections_frame1)
    tracks2 = tracker.update(1, detections_frame2)
    
    print(f"Frame 0: {len(tracks1)} tracks")
    print(f"Frame 1: {len(tracks2)} tracks")
    
    # Test trajectory analysis
    df = analyze_trajectories(tracker.get_all_tracks())
    print(f"Trajectory analysis: {len(df)} tracks")
    
    # Test anomaly detection
    anomalies = detect_tracking_anomalies(tracker.get_all_tracks(), config)
    print(f"Anomalies detected: {len(anomalies)}")
    
    print("Tracking utilities test completed")
