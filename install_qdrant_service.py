"""
Install QDrant as a macOS startup service using launchd
"""

import os
import sys
import subprocess
from pathlib import Path
import plistlib
from qdrant_utils import wait_for_qdrant


def create_launchd_plist(qdrant_binary_path: str, data_dir: str) -> str:
    """Create launchd plist for QDrant service"""
    
    plist_content = {
        'Label': 'com.myna.qdrant',
        'ProgramArguments': [
            str(qdrant_binary_path),
            '--storage-path', str(data_dir),
            '--http-port', '6333',
            '--grpc-port', '6334'
        ],
        'RunAtLoad': True,
        'KeepAlive': True,
        'StandardOutPath': str(Path.home() / 'Library/Logs/qdrant.log'),
        'StandardErrorPath': str(Path.home() / 'Library/Logs/qdrant_error.log'),
        'WorkingDirectory': str(Path.cwd()),
        'EnvironmentVariables': {
            'PATH': '/usr/local/bin:/usr/bin:/bin'
        }
    }
    
    # Write plist file
    plist_path = Path.home() / 'Library/LaunchAgents/com.myna.qdrant.plist'
    plist_path.parent.mkdir(exist_ok=True)
    
    with open(plist_path, 'wb') as f:
        plistlib.dump(plist_content, f)
    
    return str(plist_path)


def check_docker():
    """Check if Docker is available"""
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def setup_docker_service():
    """Setup QDrant using Docker with restart policy"""
    print("Setting up QDrant with Docker...")
    
    # Create data directory in user's home
    data_dir = Path.home() / ".qdrant_data"
    data_dir.mkdir(exist_ok=True)
    
    # Stop and remove existing container
    subprocess.run(["docker", "stop", "myna-qdrant"], capture_output=True)
    subprocess.run(["docker", "rm", "myna-qdrant"], capture_output=True)
    
    # Start container with restart policy
    cmd = [
        "docker", "run", "-d",
        "--name", "myna-qdrant",
        "--restart", "unless-stopped",  # Auto-restart on boot
        "-p", "6333:6333", 
        "-p", "6334:6334",
        "-v", f"{data_dir.absolute()}:/qdrant/storage:z",
        "qdrant/qdrant:latest"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ QDrant Docker container started with auto-restart: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to start QDrant container: {e.stderr}")
        return False


def download_qdrant_binary():
    """Download QDrant binary for local execution"""
    import platform
    import urllib.request
    
    system = platform.system().lower()
    arch = platform.machine().lower()
    
    # Map to QDrant release naming
    if system == "darwin":
        if arch in ["arm64", "aarch64"]:
            binary_name = "qdrant-aarch64-apple-darwin"
        else:
            binary_name = "qdrant-x86_64-apple-darwin"
    else:
        print(f"Binary installation only supported on macOS, got: {system} {arch}")
        return None
    
    print(f"Downloading QDrant binary for {system} {arch}...")
    
    # Get latest release
    url = f"https://github.com/qdrant/qdrant/releases/latest/download/{binary_name}"
    binary_path = Path.cwd() / "qdrant"
    
    try:
        urllib.request.urlretrieve(url, binary_path)
        binary_path.chmod(0o755)
        print(f"✓ Downloaded QDrant binary to {binary_path}")
        return binary_path
    except Exception as e:
        print(f"✗ Failed to download QDrant binary: {e}")
        return None


def setup_binary_service(binary_path: Path):
    """Setup QDrant binary as launchd service"""
    print("Setting up QDrant as macOS service...")
    
    # Create data directory in user's home
    data_dir = Path.home() / ".qdrant_data"
    data_dir.mkdir(exist_ok=True)
    
    # Create launchd plist
    plist_path = create_launchd_plist(str(binary_path), str(data_dir))
    print(f"✓ Created launchd plist: {plist_path}")
    
    # Load the service
    try:
        subprocess.run(["launchctl", "unload", plist_path], capture_output=True)  # Unload if exists
        subprocess.run(["launchctl", "load", plist_path], check=True, capture_output=True)
        print("✓ QDrant service loaded and will start on boot")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to load QDrant service: {e}")
        return False




def uninstall_service():
    """Uninstall QDrant service"""
    print("Uninstalling QDrant service...")
    
    # Stop Docker container
    try:
        subprocess.run(["docker", "stop", "myna-qdrant"], check=True, capture_output=True)
        subprocess.run(["docker", "rm", "myna-qdrant"], capture_output=True)
        print("✓ Stopped and removed Docker container")
    except subprocess.CalledProcessError:
        pass
    
    # Unload launchd service
    plist_path = Path.home() / 'Library/LaunchAgents/com.myna.qdrant.plist'
    if plist_path.exists():
        try:
            subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
            plist_path.unlink()
            print("✓ Removed launchd service")
        except Exception as e:
            print(f"Warning: Could not remove launchd service: {e}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "uninstall":
        uninstall_service()
        return
    
    print("Installing QDrant as startup service...")
    
    # Try Docker first (easier, more reliable)
    if check_docker():
        print("Docker detected, using Docker for QDrant service")
        if setup_docker_service():
            if wait_for_qdrant():
                print("\n✅ QDrant is installed and running!")
                print("🔄 It will automatically start on system boot")
                print("🌐 Available at: http://localhost:6333")
                print("\nTo uninstall: python install_qdrant_service.py uninstall")
                return
    
    # Fallback to binary + launchd (macOS only)
    print("Docker not available, trying binary installation...")
    binary_path = download_qdrant_binary()
    if binary_path:
        if setup_binary_service(binary_path):
            if wait_for_qdrant():
                print("\n✅ QDrant is installed and running!")
                print("🔄 It will automatically start on system boot")
                print("🌐 Available at: http://localhost:6333")
                print("\nTo uninstall: python install_qdrant_service.py uninstall")
                return
    
    print("\n❌ Failed to install QDrant service")
    print("Manual options:")
    print("1. Install Docker and try again")
    print("2. Install QDrant manually: https://qdrant.tech/documentation/quick-start/")
    print("3. Use QDrant Cloud: https://cloud.qdrant.io/")


if __name__ == "__main__":
    main()