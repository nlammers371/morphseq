#!/usr/bin/env python
"""
Configuration utilities for the MorphSeq embryo segmentation pipeline.
Handles loading, validation, and access to configuration parameters.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class PipelineConfig:
    """
    Configuration manager for the embryo segmentation pipeline.
    
    Loads and validates configuration from YAML file, provides convenient
    access to configuration parameters, and handles path resolution.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration YAML file. If None, uses default.
        """
        if config_path is None:
            # Default to config in same directory as this script
            config_dir = Path(__file__).parent.parent / "configs"
            config_path = config_dir / "pipeline_config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        self._resolve_paths()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _validate_config(self):
        """Validate that required configuration sections exist."""
        required_sections = ['paths', 'models', 'detection', 'qc', 'processing', 'coco']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Required configuration section missing: {section}")
    
    def _resolve_paths(self):
        """Resolve relative paths to absolute paths."""
        # Get sandbox root directory
        sandbox_root = Path(__file__).parent.parent
        
        # Resolve paths relative to sandbox or as absolute paths
        paths = self.config['paths']
        
        # Get base morphseq data directory
        morphseq_data_dir = Path(paths.get('morphseq_data_dir', ''))
        
        # Resolve paths relative to morphseq_data_dir
        morphseq_relative_paths = [
            'stitched_images_dir', 
            'embryo_mask_root', 
            'yolk_mask_root', 
            'via_mask_root'
        ]
        
        for key in morphseq_relative_paths:
            if key in paths:
                rel_path = Path(paths[key])
                if not rel_path.is_absolute():
                    paths[key] = str(morphseq_data_dir / rel_path)
        
        # Resolve sandbox-relative output directories
        sandbox_relative_paths = [
            'intermediate_dir', 
            'final_dir', 
            'logs_dir', 
            'morphseq_well_videos',
            'models_dir',
            'groundingdino_path',
            'sam2_path'
        ]
        
        for key in sandbox_relative_paths:
            if key in paths:
                rel_path = Path(paths[key])
                if not rel_path.is_absolute():
                    paths[key] = str(sandbox_root / rel_path)
        
        # Resolve model file paths relative to their respective directories
        models = self.config['models']
        if 'groundingdino' in models:
            grounding_dir = Path(paths.get('groundingdino_path', ''))
            if 'config' in models['groundingdino']:
                config_path = Path(models['groundingdino']['config'])
                if not config_path.is_absolute():
                    models['groundingdino']['config'] = str(grounding_dir / config_path)
            if 'weights' in models['groundingdino']:
                weights_path = Path(models['groundingdino']['weights'])
                if not weights_path.is_absolute():
                    models['groundingdino']['weights'] = str(grounding_dir / weights_path)
        
        if 'sam2' in models:
            sam2_dir = Path(paths.get('sam2_path', ''))
            if 'config' in models['sam2']:
                config_path = Path(models['sam2']['config'])
                if not config_path.is_absolute():
                    models['sam2']['config'] = str(sam2_dir / config_path)
            if 'checkpoint' in models['sam2']:
                checkpoint_path = Path(models['sam2']['checkpoint'])
                if not checkpoint_path.is_absolute():
                    models['sam2']['checkpoint'] = str(sam2_dir / checkpoint_path)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value (e.g., 'detection.box_threshold')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_paths(self) -> Dict[str, str]:
        """Get all path configurations."""
        return self.config['paths'].copy()
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        return self.config['models'].get(model_name, {})
    
    def get_detection_params(self) -> Dict[str, Any]:
        """Get all detection parameters."""
        return self.config['detection'].copy()
    
    def get_qc_params(self) -> Dict[str, Any]:
        """Get all quality control parameters."""
        return self.config['qc'].copy()
    
    def get_processing_params(self) -> Dict[str, Any]:
        """Get all processing parameters."""
        return self.config['processing'].copy()
    
    def get_coco_categories(self) -> list:
        """Get COCO dataset categories."""
        return self.config['coco']['categories'].copy()
    
    def ensure_output_dirs(self):
        """Ensure all output directories exist."""
        paths = self.get_paths()
        
        for dir_key in ['intermediate_dir', 'final_dir', 'logs_dir']:
            if dir_key in paths:
                dir_path = Path(paths[dir_key])
                dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_intermediate_path(self, filename: str) -> str:
        """Get path for intermediate file."""
        return os.path.join(self.get('paths.intermediate_dir'), filename)
    
    def get_final_path(self, filename: str) -> str:
        """Get path for final output file."""
        return os.path.join(self.get('paths.final_dir'), filename)
    
    def get_log_path(self, filename: str) -> str:
        """Get path for log file."""
        return os.path.join(self.get('paths.logs_dir'), filename)
    
    def get_morphseq_data_path(self, subpath: str = '') -> str:
        """
        Get path within the morphseq data directory.
        
        Args:
            subpath: Relative path within morphseq data directory
            
        Returns:
            Full path to morphseq data location
        """
        morphseq_dir = self.get('paths.morphseq_data_dir')
        if subpath:
            return os.path.join(morphseq_dir, subpath)
        return morphseq_dir
    
    def get_stitched_images_dir(self) -> str:
        """Get full path to stitched images directory."""
        return self.get('paths.stitched_images_dir')
    
    def get_mask_dir(self, mask_type: str) -> str:
        """
        Get full path to mask directory.
        
        Args:
            mask_type: Type of mask ('embryo', 'yolk', or 'via')
            
        Returns:
            Full path to mask directory
        """
        mask_key_map = {
            'embryo': 'embryo_mask_root',
            'yolk': 'yolk_mask_root', 
            'via': 'via_mask_root'
        }
        
        if mask_type not in mask_key_map:
            raise ValueError(f"Unknown mask type: {mask_type}. Must be one of: {list(mask_key_map.keys())}")
        
        return self.get(f'paths.{mask_key_map[mask_type]}')
    
    def get_model_dir(self, model_name: str = '') -> str:
        """
        Get full path to model directory.
        
        Args:
            model_name: Specific model subdirectory (optional)
            
        Returns:
            Full path to model directory
        """
        models_dir = self.get('paths.models_dir')
        if model_name:
            return os.path.join(models_dir, model_name)
        return models_dir


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """
    Convenience function to load pipeline configuration.
    
    Args:
        config_path: Path to configuration file. If None, uses default.
        
    Returns:
        PipelineConfig instance
    """
    return PipelineConfig(config_path)


# Example usage and testing
if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    
    print("Configuration loaded successfully!")
    print(f"Box threshold: {config.get('detection.box_threshold')}")
    print(f"Intermediate dir: {config.get('paths.intermediate_dir')}")
    print(f"Model config: {config.get_model_config('groundingdino')}")
    
    # Test morphseq data path utilities
    print(f"\nMorphSeq data paths:")
    print(f"Base morphseq dir: {config.get_morphseq_data_path()}")
    print(f"Stitched images: {config.get_stitched_images_dir()}")
    print(f"Embryo masks: {config.get_mask_dir('embryo')}")
    print(f"Yolk masks: {config.get_mask_dir('yolk')}")
    print(f"VIA masks: {config.get_mask_dir('via')}")
    
    print(f"\nModel paths:")
    print(f"Models dir: {config.get_model_dir()}")
    print(f"GroundingDINO dir: {config.get('paths.groundingdino_path')}")
    print(f"SAM2 dir: {config.get('paths.sam2_path')}")
    
    print(f"\nExample subpath: {config.get_morphseq_data_path('custom_data/experiment1')}")
    
    # Ensure output directories exist
    config.ensure_output_dirs()
    print("Output directories created/verified")
