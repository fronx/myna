"""
QDrant utility functions
"""

import os
import shutil
import subprocess
from pathlib import Path
import requests


def is_qdrant_running(url: str) -> bool:
    """Check if QDrant is running at the given URL"""
    try:
        response = requests.get(url, timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def wait_for_qdrant(url: str = "http://localhost:6333", timeout: int = 30) -> bool:
    """Wait for QDrant to be ready"""
    print("Waiting for QDrant to be ready...")
    
    for i in range(timeout):
        if is_qdrant_running(url):
            print("✓ QDrant is running and ready!")
            return True
        
        import time
        time.sleep(1)
        if i % 5 == 0 and i > 0:
            print(f"Still waiting... ({i}s)")
    
    print("✗ QDrant failed to start within timeout")
    return False


def migrate_qdrant_data() -> bool:
    """
    Migrate QDrant data from local project directory to global ~/.qdrant_data
    Returns True if migration was performed, False if no migration needed
    """
    old_data_dir = Path.cwd() / "qdrant_data"
    new_data_dir = Path.home() / ".qdrant_data"
    
    if not old_data_dir.exists():
        print("No local qdrant_data directory found - no migration needed")
        return False
    
    if new_data_dir.exists() and any(new_data_dir.iterdir()):
        print(f"Global QDrant data directory already exists at {new_data_dir}")
        print(f"Local data found at {old_data_dir}")
        
        response = input("Do you want to migrate local data to global location? (y/n): ").lower().strip()
        if response not in ['y', 'yes']:
            print("Skipping migration")
            return False
    
    print(f"Migrating QDrant data from {old_data_dir} to {new_data_dir}")
    
    try:
        # Stop QDrant if running to avoid data corruption
        subprocess.run(["docker", "stop", "myna-qdrant"], capture_output=True)
        
        # Create new directory
        new_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy data
        if old_data_dir.exists():
            for item in old_data_dir.iterdir():
                if item.is_dir():
                    shutil.copytree(item, new_data_dir / item.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, new_data_dir / item.name)
        
        # Remove old directory
        shutil.rmtree(old_data_dir)
        
        print("✓ Migration completed successfully!")
        print(f"QDrant data is now at: {new_data_dir}")
        print("Note: QDrant will be restarted automatically with the new data location")
        
        return True
        
    except Exception as e:
        print(f"✗ Migration failed: {e}")
        return False