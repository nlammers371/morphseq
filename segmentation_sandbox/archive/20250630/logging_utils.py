#!/usr/bin/env python
"""
Logging utilities for the MorphSeq embryo segmentation pipeline.
Provides structured logging with different levels and output formats.
"""

import os
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import sys


class PipelineLogger:
    """
    Structured logger for the segmentation pipeline.
    Handles both console and file logging with different levels.
    """
    
    def __init__(self, name: str = "morphseq_pipeline", 
                 log_dir: Optional[Union[str, Path]] = None,
                 log_level: str = "INFO",
                 log_to_file: bool = True,
                 log_to_console: bool = True):
        """
        Initialize pipeline logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_to_file: Whether to log to file
            log_to_console: Whether to log to console
        """
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "logs"
        self.log_level = getattr(logging, log_level.upper())
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        
        # Ensure log directory exists
        if self.log_to_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set up handlers
        self._setup_handlers()
        
        # Track pipeline state
        self.pipeline_start_time = datetime.now()
        self.current_stage = None
        self.stage_start_time = None
        
    def _setup_handlers(self):
        """Set up logging handlers."""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        if self.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"{self.name}_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            self.log_file_path = log_file
    
    def start_pipeline(self, config: Optional[Dict[str, Any]] = None):
        """Log pipeline start."""
        self.pipeline_start_time = datetime.now()
        self.logger.info("="*60)
        self.logger.info("MorphSeq Embryo Segmentation Pipeline Started")
        self.logger.info(f"Start time: {self.pipeline_start_time}")
        
        if config:
            self.logger.info("Pipeline configuration:")
            for key, value in config.items():
                if isinstance(value, dict):
                    self.logger.info(f"  {key}:")
                    for subkey, subvalue in value.items():
                        self.logger.info(f"    {subkey}: {subvalue}")
                else:
                    self.logger.info(f"  {key}: {value}")
        
        self.logger.info("="*60)
    
    def start_stage(self, stage_name: str, **kwargs):
        """Log stage start."""
        self.current_stage = stage_name
        self.stage_start_time = datetime.now()
        
        self.logger.info(f"\nStarting stage: {stage_name}")
        if kwargs:
            for key, value in kwargs.items():
                self.logger.info(f"  {key}: {value}")
    
    def end_stage(self, stage_name: str, **kwargs):
        """Log stage completion."""
        if self.stage_start_time:
            duration = datetime.now() - self.stage_start_time
            self.logger.info(f"Completed stage: {stage_name}")
            self.logger.info(f"Stage duration: {duration}")
            
            if kwargs:
                for key, value in kwargs.items():
                    self.logger.info(f"  {key}: {value}")
        
        self.current_stage = None
        self.stage_start_time = None
    
    def end_pipeline(self, **kwargs):
        """Log pipeline completion."""
        duration = datetime.now() - self.pipeline_start_time
        self.logger.info("="*60)
        self.logger.info("MorphSeq Embryo Segmentation Pipeline Completed")
        self.logger.info(f"Total duration: {duration}")
        
        if kwargs:
            self.logger.info("Pipeline summary:")
            for key, value in kwargs.items():
                self.logger.info(f"  {key}: {value}")
        
        self.logger.info("="*60)
    
    def log_video_processing(self, video_path: str, frame_count: int, **kwargs):
        """Log video processing information."""
        self.logger.info(f"Processing video: {video_path}")
        self.logger.info(f"Frame count: {frame_count}")
        
        for key, value in kwargs.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_detection_results(self, frame_idx: int, detection_count: int, 
                            detection_stats: Optional[Dict] = None):
        """Log detection results for a frame."""
        self.logger.debug(f"Frame {frame_idx}: {detection_count} detections")
        
        if detection_stats and self.logger.isEnabledFor(logging.DEBUG):
            for key, value in detection_stats.items():
                if isinstance(value, float):
                    self.logger.debug(f"  {key}: {value:.3f}")
                else:
                    self.logger.debug(f"  {key}: {value}")
    
    def log_flag(self, flag_type: str, message: str, video_path: str = None, 
                frame_idx: int = None, **kwargs):
        """Log a quality control flag."""
        log_msg = f"FLAG [{flag_type}]: {message}"
        
        if video_path:
            log_msg += f" (Video: {video_path})"
        if frame_idx is not None:
            log_msg += f" (Frame: {frame_idx})"
        
        self.logger.warning(log_msg)
        
        if kwargs:
            for key, value in kwargs.items():
                self.logger.warning(f"  {key}: {value}")
    
    def log_error(self, error_msg: str, exception: Optional[Exception] = None, **kwargs):
        """Log an error with context."""
        self.logger.error(f"ERROR: {error_msg}")
        
        if exception:
            self.logger.error(f"Exception: {str(exception)}")
        
        if kwargs:
            for key, value in kwargs.items():
                self.logger.error(f"  {key}: {value}")
    
    def log_intermediate_save(self, file_path: str, data_type: str, **kwargs):
        """Log intermediate file save."""
        self.logger.debug(f"Saved {data_type} to: {file_path}")
        
        for key, value in kwargs.items():
            self.logger.debug(f"  {key}: {value}")
    
    def log_config_info(self, config_section: str, config_data: Dict[str, Any]):
        """Log configuration information."""
        self.logger.info(f"Configuration - {config_section}:")
        for key, value in config_data.items():
            self.logger.info(f"  {key}: {value}")
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)


class QCLogger:
    """
    Quality control logger for tracking flags and issues.
    Maintains structured records of all QC flags for analysis.
    """
    
    def __init__(self, log_dir: Union[str, Path]):
        """
        Initialize QC logger.
        
        Args:
            log_dir: Directory for QC log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize QC records
        self.video_flags = {}  # video_path -> list of flags
        self.frame_flags = {}  # (video_path, frame_idx) -> list of flags
        self.global_flags = []  # Global pipeline flags
        
        self.qc_file = self.log_dir / f"qc_flags_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    def add_video_flag(self, video_path: str, flag_type: str, message: str, **kwargs):
        """Add a video-level QC flag."""
        if video_path not in self.video_flags:
            self.video_flags[video_path] = []
        
        flag = {
            'timestamp': datetime.now().isoformat(),
            'flag_type': flag_type,
            'message': message,
            'metadata': kwargs
        }
        
        self.video_flags[video_path].append(flag)
    
    def add_frame_flag(self, video_path: str, frame_idx: int, flag_type: str, 
                      message: str, **kwargs):
        """Add a frame-level QC flag."""
        key = (video_path, frame_idx)
        if key not in self.frame_flags:
            self.frame_flags[key] = []
        
        flag = {
            'timestamp': datetime.now().isoformat(),
            'flag_type': flag_type,
            'message': message,
            'metadata': kwargs
        }
        
        self.frame_flags[key].append(flag)
    
    def add_global_flag(self, flag_type: str, message: str, **kwargs):
        """Add a global pipeline flag."""
        flag = {
            'timestamp': datetime.now().isoformat(),
            'flag_type': flag_type,
            'message': message,
            'metadata': kwargs
        }
        
        self.global_flags.append(flag)
    
    def get_video_flags(self, video_path: str) -> List[Dict]:
        """Get all flags for a specific video."""
        return self.video_flags.get(video_path, [])
    
    def get_frame_flags(self, video_path: str, frame_idx: int) -> List[Dict]:
        """Get all flags for a specific frame."""
        return self.frame_flags.get((video_path, frame_idx), [])
    
    def get_flag_summary(self) -> Dict[str, Any]:
        """Get summary of all flags."""
        summary = {
            'video_flags_count': sum(len(flags) for flags in self.video_flags.values()),
            'frame_flags_count': sum(len(flags) for flags in self.frame_flags.values()),
            'global_flags_count': len(self.global_flags),
            'videos_with_flags': len(self.video_flags),
            'frames_with_flags': len(self.frame_flags)
        }
        
        # Count by flag type
        flag_type_counts = {}
        
        for flags in self.video_flags.values():
            for flag in flags:
                flag_type = flag['flag_type']
                flag_type_counts[flag_type] = flag_type_counts.get(flag_type, 0) + 1
        
        for flags in self.frame_flags.values():
            for flag in flags:
                flag_type = flag['flag_type']
                flag_type_counts[flag_type] = flag_type_counts.get(flag_type, 0) + 1
        
        for flag in self.global_flags:
            flag_type = flag['flag_type']
            flag_type_counts[flag_type] = flag_type_counts.get(flag_type, 0) + 1
        
        summary['flag_type_counts'] = flag_type_counts
        
        return summary
    
    def save_qc_report(self):
        """Save QC report to file."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_flag_summary(),
            'video_flags': self.video_flags,
            'frame_flags': {f"{k[0]}_{k[1]}": v for k, v in self.frame_flags.items()},
            'global_flags': self.global_flags
        }
        
        try:
            with open(self.qc_file, 'w') as f:
                json.dump(report, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving QC report: {e}")
            return False


def setup_pipeline_logging(config: Dict[str, Any], stage_name: str = None) -> PipelineLogger:
    """
    Set up pipeline logging from configuration.
    
    Args:
        config: Pipeline configuration
        stage_name: Optional stage name for logger
        
    Returns:
        Configured PipelineLogger
    """
    logging_config = config.get('logging', {})
    paths_config = config.get('paths', {})
    
    logger_name = f"morphseq_pipeline"
    if stage_name:
        logger_name += f"_{stage_name}"
    
    logger = PipelineLogger(
        name=logger_name,
        log_dir=paths_config.get('logs_dir', 'logs'),
        log_level=logging_config.get('level', 'INFO'),
        log_to_file=logging_config.get('log_to_file', True),
        log_to_console=logging_config.get('log_to_console', True)
    )
    
    return logger


# Example usage and testing
if __name__ == "__main__":
    # Test logging utilities
    print("Testing logging utilities...")
    
    # Create test logger
    logger = PipelineLogger("test_pipeline", log_dir="test_logs")
    
    # Test pipeline logging
    logger.start_pipeline({"test_param": "test_value"})
    
    logger.start_stage("test_stage", input_files=5)
    logger.info("Processing test data...")
    logger.log_flag("HIGH_VARIANCE", "Test flag", video_path="test.mp4")
    logger.end_stage("test_stage", processed_files=5)
    
    logger.end_pipeline(total_processed=5)
    
    # Test QC logger
    qc_logger = QCLogger("test_logs")
    qc_logger.add_video_flag("test.mp4", "MISSING_FRAMES", "Some frames missing")
    qc_logger.add_frame_flag("test.mp4", 10, "LOW_QUALITY", "Poor detection quality")
    
    summary = qc_logger.get_flag_summary()
    print(f"QC Summary: {summary}")
    
    qc_logger.save_qc_report()
    
    print("Logging utilities test completed")
