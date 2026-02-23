"""
Base File Handler for MorphSeq Pipeline
Handles JSON I/O, backups, and change tracking
"""

import json
import shutil
from datetime import datetime
from pathlib import Path 
from typing import Dict, List, Union, Optional

class BaseFileHandler:
    """
    Base class for file I/O operations and utilities.
    
    Provides:
    - JSON load/save with atomic writes
    - Backup management
    - Change tracking
    - Timestamps
    """
    
    def __init__(self, filepath: Union[str, Path], verbose: bool = True):
        self.filepath = Path(filepath)
        self.verbose = verbose
        self._unsaved_changes = False
        self._change_log = []
    
    # File I/O Operations
    def load_json(self, file_path: Path = None) -> Dict:
        """Load JSON file with error handling."""
        if file_path is None:
            file_path = self.filepath
            
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")
    
    def save_json(self, data: Dict, file_path: Path = None, 
                  create_backup: bool = True) -> None:
        """Save JSON with atomic write and optional backup."""
        if file_path is None:
            file_path = self.filepath
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if create_backup and file_path.exists():
            self._create_backup(file_path)
        
        # Atomic write
        temp_path = file_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            temp_path.replace(file_path)
            self._unsaved_changes = False
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def _create_backup(self, file_path: Path) -> Path:
        """Create timestamped backup and clean up old ones."""
        # Create new backup first
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.with_suffix(f'.backup.{timestamp}{file_path.suffix}')
        shutil.copy2(file_path, backup_path)
        if self.verbose:
            print(f"📦 Created backup: {backup_path.name}")
        
        # Then clean up old backups
        self._cleanup_old_backups(file_path)
        
        return backup_path
    
    def _cleanup_old_backups(self, file_path: Path, keep_count: int = 1) -> None:
        """Remove old backup files, keeping only the most recent ones."""
        backup_pattern = f"{file_path.stem}.backup.*{file_path.suffix}"
        backup_files = sorted(
            file_path.parent.glob(backup_pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True  # Most recent first
        )
        
        # Remove old backups beyond keep_count
        for old_backup in backup_files[keep_count:]:
            try:
                old_backup.unlink()
                if self.verbose:
                    print(f"🗑️  Removed old backup: {old_backup.name}")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Could not remove backup {old_backup.name}: {e}")
    
    def cleanup_backups(self, keep_count: int = 1) -> None:
        """Manually clean up backup files, keeping only the most recent ones."""
        self._cleanup_old_backups(self.filepath, keep_count)
    
    # Change tracking
    def log_operation(self, operation: str, entity_id: str = None, **kwargs):
        """Log an operation."""
        self._change_log.append({
            "timestamp": self.get_timestamp(),
            "operation": operation,
            "entity_id": entity_id,
            **kwargs
        })
        self._unsaved_changes = True
        
        if self.verbose:
            print(f"📝 {operation}: {entity_id or 'N/A'}")
    
    # Utilities
    def get_timestamp(self) -> str:
        """Get current ISO timestamp."""
        return datetime.now().isoformat()
    
    @property
    def has_unsaved_changes(self) -> bool:
        return self._unsaved_changes
    
    def get_file_stats(self) -> Dict:
        """Get file statistics."""
        if not self.filepath.exists():
            return {"exists": False}
        
        stat = self.filepath.stat()
        return {
            "exists": True,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
        }
