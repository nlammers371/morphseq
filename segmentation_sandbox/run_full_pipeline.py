#!/usr/bin/env python
"""
run_full_pipeline.py

Orchestrator script for the complete MorphSeq embryo segmentation pipeline.
Runs all stages in sequence with proper error handling and progress reporting.

This script:
1. Validates configuration and model availability
2. Runs all pipeline stages in sequence
3. Provides progress reporting and error handling
4. Generates consolidated reports and logs
5. Allows selective stage execution and resume functionality

Usage:
    python run_full_pipeline.py [--config CONFIG_PATH] [--start_stage STAGE] [--end_stage STAGE] [--stages STAGE1,STAGE2,...]
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import load_config, setup_pipeline_logging, QCLogger


class PipelineOrchestrator:
    """
    Orchestrates the complete MorphSeq embryo segmentation pipeline.
    """
    
    def __init__(self, config):
        """Initialize pipeline orchestrator."""
        self.config = config
        self.logger = setup_pipeline_logging(config.config, "orchestrator")
        self.qc_logger = QCLogger(config.get('paths.logs_dir'))
        
        # Define pipeline stages
        self.stages = [
            {
                'name': 'prepare_videos',
                'script': '01_prepare_videos.py',
                'description': 'Convert stitched images to videos',
                'required_inputs': ['stitched_images_dir'],
                'outputs': ['morphseq_well_videos']
            },
            {
                'name': 'initial_detection',
                'script': '02_initial_detection.py',
                'description': 'Run GroundingDINO detection on all frames',
                'required_inputs': ['morphseq_well_videos'],
                'outputs': ['detections']
            },
            {
                'name': 'generate_masks',
                'script': '03_generate_sam2_masks.py',
                'description': 'Generate SAM2 masks for detections',
                'required_inputs': ['detections'],
                'outputs': ['masks']
            },
            {
                'name': 'track_embryos',
                'script': '04_track_embryos.py',
                'description': 'Track embryos across frames',
                'required_inputs': ['masks'],
                'outputs': ['tracks']
            },
            {
                'name': 'generate_coco',
                'script': '05_generate_coco_annotations.py',
                'description': 'Generate final COCO annotations',
                'required_inputs': ['tracks'],
                'outputs': ['final']
            }
        ]
        
        # Pipeline state
        self.pipeline_start_time = None
        self.stage_results = {}
        self.overall_success = True
        
        # Scripts directory
        self.scripts_dir = Path(__file__).parent
    
    def validate_configuration(self) -> bool:
        """Validate pipeline configuration and dependencies."""
        self.logger.info("Validating pipeline configuration...")
        
        validation_passed = True
        
        # Check required paths
        required_paths = [
            'morphseq_data_dir',
            'stitched_images_dir',
            'embryo_mask_root',
            'models_dir'
        ]
        
        for path_key in required_paths:
            if path_key == 'morphseq_data_dir':
                path_value = self.config.get_morphseq_data_path()
            elif path_key == 'stitched_images_dir':
                path_value = self.config.get_stitched_images_dir()
            elif path_key == 'embryo_mask_root':
                path_value = self.config.get_mask_dir('embryo')
            elif path_key == 'models_dir':
                path_value = self.config.get_model_dir()
            else:
                path_value = self.config.get(f'paths.{path_key}')
            
            if not path_value:
                self.logger.error(f"Required path not configured: {path_key}")
                validation_passed = False
                continue
            
            path_obj = Path(path_value)
            if not path_obj.exists():
                self.logger.warning(f"Path does not exist: {path_key} = {path_value}")
                if path_key in ['stitched_images_dir', 'embryo_mask_root']:
                    self.logger.error(f"Required input path missing: {path_key}")
                    validation_passed = False
        
        # Check model configurations
        model_configs = ['groundingdino', 'sam2']
        for model_name in model_configs:
            model_config = self.config.get_model_config(model_name)
            if not model_config:
                self.logger.error(f"Model configuration missing: {model_name}")
                validation_passed = False
                continue
            
            # Check model files (warn only, as they might be downloaded later)
            config_file = model_config.get('config')
            if config_file and not Path(config_file).exists():
                self.logger.warning(f"{model_name} config file not found: {config_file}")
            
            if model_name == 'groundingdino':
                weights_file = model_config.get('weights')
                if weights_file and not Path(weights_file).exists():
                    self.logger.warning(f"{model_name} weights file not found: {weights_file}")
            elif model_name == 'sam2':
                checkpoint_file = model_config.get('checkpoint')
                if checkpoint_file and not Path(checkpoint_file).exists():
                    self.logger.warning(f"{model_name} checkpoint file not found: {checkpoint_file}")
        
        # Check required scripts
        for stage in self.stages:
            script_path = self.scripts_dir / stage['script']
            if not script_path.exists():
                self.logger.error(f"Pipeline script not found: {script_path}")
                validation_passed = False
        
        if validation_passed:
            self.logger.info("Configuration validation passed")
        else:
            self.logger.error("Configuration validation failed")
            self.qc_logger.add_global_flag(
                "CONFIGURATION_VALIDATION_FAILED",
                "Pipeline configuration validation failed"
            )
        
        return validation_passed
    
    def check_stage_prerequisites(self, stage: Dict[str, Any]) -> bool:
        """Check if stage prerequisites are met."""
        stage_name = stage['name']
        
        # Check required inputs exist
        for input_name in stage['required_inputs']:
            if input_name == 'stitched_images_dir':
                input_path = self.config.get_stitched_images_dir()
            elif input_name == 'morphseq_well_videos':
                input_path = self.config.get('paths.morphseq_well_videos')
            else:
                input_path = self.config.get_intermediate_path(input_name)
            
            if not Path(input_path).exists():
                self.logger.error(f"Stage {stage_name} prerequisite not met: {input_name} not found at {input_path}")
                return False
        
        return True
    
    def run_stage(self, stage: Dict[str, Any]) -> bool:
        """Run a single pipeline stage."""
        stage_name = stage['name']
        script_name = stage['script']
        description = stage['description']
        
        self.logger.info(f"Starting stage: {stage_name}")
        self.logger.info(f"Description: {description}")
        
        # Check prerequisites
        if not self.check_stage_prerequisites(stage):
            self.logger.error(f"Prerequisites not met for stage: {stage_name}")
            return False
        
        # Prepare command
        script_path = self.scripts_dir / script_name
        config_path = self.config.config_path
        
        import subprocess
        cmd = [sys.executable, str(script_path), '--config', str(config_path)]
        
        self.logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run stage
        stage_start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.scripts_dir.parent  # Run from sandbox root
            )
            
            stage_duration = time.time() - stage_start_time
            
            # Log results
            if result.returncode == 0:
                self.logger.info(f"Stage {stage_name} completed successfully")
                self.logger.info(f"Duration: {stage_duration:.1f} seconds")
                
                # Log stdout if available
                if result.stdout:
                    self.logger.debug(f"Stage stdout: {result.stdout}")
                
                self.stage_results[stage_name] = {
                    'success': True,
                    'duration': stage_duration,
                    'return_code': result.returncode
                }
                
                return True
                
            else:
                self.logger.error(f"Stage {stage_name} failed with return code {result.returncode}")
                
                # Log stderr
                if result.stderr:
                    self.logger.error(f"Stage stderr: {result.stderr}")
                
                self.stage_results[stage_name] = {
                    'success': False,
                    'duration': stage_duration,
                    'return_code': result.returncode,
                    'error': result.stderr
                }
                
                self.qc_logger.add_global_flag(
                    "STAGE_EXECUTION_FAILED",
                    f"Stage {stage_name} failed with return code {result.returncode}",
                    stage=stage_name,
                    error=result.stderr
                )
                
                return False
                
        except Exception as e:
            stage_duration = time.time() - stage_start_time
            self.logger.error(f"Error running stage {stage_name}: {e}")
            
            self.stage_results[stage_name] = {
                'success': False,
                'duration': stage_duration,
                'error': str(e)
            }
            
            return False
    
    def run_pipeline(self, stages_to_run: Optional[List[str]] = None,
                    start_stage: Optional[str] = None,
                    end_stage: Optional[str] = None) -> bool:
        """
        Run the complete pipeline or selected stages.
        
        Args:
            stages_to_run: Specific stages to run (if None, run all)
            start_stage: Stage to start from (inclusive)
            end_stage: Stage to end at (inclusive)
            
        Returns:
            True if all stages succeeded, False otherwise
        """
        self.pipeline_start_time = time.time()
        
        self.logger.start_pipeline({
            'stages_to_run': stages_to_run,
            'start_stage': start_stage,
            'end_stage': end_stage,
            'total_stages': len(self.stages)
        })
        
        # Validate configuration first
        if not self.validate_configuration():
            self.logger.error("Configuration validation failed, aborting pipeline")
            return False
        
        # Determine stages to run
        if stages_to_run:
            # Run specific stages
            selected_stages = [s for s in self.stages if s['name'] in stages_to_run]
        else:
            # Run range of stages
            selected_stages = self.stages
            
            if start_stage:
                start_idx = next((i for i, s in enumerate(self.stages) if s['name'] == start_stage), 0)
                selected_stages = selected_stages[start_idx:]
            
            if end_stage:
                end_idx = next((i for i, s in enumerate(self.stages) if s['name'] == end_stage), len(self.stages)-1)
                selected_stages = selected_stages[:end_idx+1]
        
        if not selected_stages:
            self.logger.error("No stages selected to run")
            return False
        
        self.logger.info(f"Running {len(selected_stages)} stages: {[s['name'] for s in selected_stages]}")
        
        # Run stages
        successful_stages = 0
        
        for i, stage in enumerate(selected_stages):
            self.logger.info(f"Stage {i+1}/{len(selected_stages)}: {stage['name']}")
            
            if self.run_stage(stage):
                successful_stages += 1
            else:
                self.overall_success = False
                
                # Option to continue on error or stop
                if self.config.get('processing.stop_on_error', True):
                    self.logger.error("Stopping pipeline due to stage failure")
                    break
                else:
                    self.logger.warning("Continuing pipeline despite stage failure")
        
        # Generate final report
        pipeline_duration = time.time() - self.pipeline_start_time
        
        pipeline_summary = {
            'total_stages': len(selected_stages),
            'successful_stages': successful_stages,
            'failed_stages': len(selected_stages) - successful_stages,
            'overall_success': self.overall_success,
            'total_duration': pipeline_duration,
            'stage_results': self.stage_results
        }
        
        # Save pipeline report
        report_file = Path(self.config.get('paths.final_dir')) / 'pipeline_report.json'
        from utils.file_utils import save_json
        save_json(pipeline_summary, report_file)
        
        # Save QC report
        self.qc_logger.save_qc_report()
        
        self.logger.end_pipeline(pipeline_summary)
        
        if self.overall_success:
            self.logger.info("Pipeline completed successfully!")
        else:
            self.logger.error("Pipeline completed with errors")
        
        return self.overall_success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run the complete MorphSeq embryo segmentation pipeline")
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--start_stage', type=str, default=None,
                       help='Stage to start from (prepare_videos, initial_detection, generate_masks, track_embryos, generate_coco)')
    parser.add_argument('--end_stage', type=str, default=None,
                       help='Stage to end at (prepare_videos, initial_detection, generate_masks, track_embryos, generate_coco)')
    parser.add_argument('--stages', type=str, default=None,
                       help='Comma-separated list of specific stages to run')
    parser.add_argument('--validate_only', action='store_true',
                       help='Only validate configuration, do not run pipeline')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(config)
    
    # Validate only mode
    if args.validate_only:
        if orchestrator.validate_configuration():
            print("Configuration validation passed")
            return 0
        else:
            print("Configuration validation failed")
            return 1
    
    # Parse stages argument
    stages_to_run = None
    if args.stages:
        stages_to_run = [s.strip() for s in args.stages.split(',')]
    
    # Run pipeline
    success = orchestrator.run_pipeline(
        stages_to_run=stages_to_run,
        start_stage=args.start_stage,
        end_stage=args.end_stage
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
