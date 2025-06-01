"""
QDrant setup script for local development
"""

import subprocess
import sys
import os
from pathlib import Path
from qdrant_utils import wait_for_qdrant


def check_docker():
    """Check if Docker is available"""
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def start_qdrant_docker():
    """Start QDrant using Docker"""
    print("Starting QDrant with Docker...")
    
    # Create data directory
    data_dir = Path.cwd() / "qdrant_data"
    data_dir.mkdir(exist_ok=True)
    
    cmd = [
        "docker", "run", "-d",
        "--name", "myna-qdrant",
        "-p", "6333:6333", 
        "-p", "6334:6334",
        "-v", f"{data_dir.absolute()}:/qdrant/storage:z",
        "qdrant/qdrant:latest"
    ]
    
    try:
        # Stop existing container if running
        subprocess.run(["docker", "stop", "myna-qdrant"], capture_output=True)
        subprocess.run(["docker", "rm", "myna-qdrant"], capture_output=True)
        
        # Start new container
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"QDrant container started: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to start QDrant container: {e.stderr}")
        return False


def download_qdrant_binary():
    """Download QDrant binary for local execution"""
    import platform
    
    system = platform.system().lower()
    arch = platform.machine().lower()
    
    # Map to QDrant release naming
    if system == "darwin":
        if arch in ["arm64", "aarch64"]:
            binary_name = "qdrant-aarch64-apple-darwin"
        else:
            binary_name = "qdrant-x86_64-apple-darwin"
    elif system == "linux":
        if arch in ["arm64", "aarch64"]:
            binary_name = "qdrant-aarch64-unknown-linux-gnu"
        else:
            binary_name = "qdrant-x86_64-unknown-linux-gnu"
    else:
        print(f"Unsupported platform: {system} {arch}")
        return None
    
    print(f"Downloading QDrant binary for {system} {arch}...")
    
    # Get latest release
    url = f"https://github.com/qdrant/qdrant/releases/latest/download/{binary_name}"
    binary_path = Path.cwd() / "qdrant"
    
    try:
        import urllib.request
        urllib.request.urlretrieve(url, binary_path)
        binary_path.chmod(0o755)
        print(f"Downloaded QDrant binary to {binary_path}")
        return binary_path
    except Exception as e:
        print(f"Failed to download QDrant binary: {e}")
        return None


def start_qdrant_binary(binary_path: Path):
    """Start QDrant using downloaded binary"""
    print("Starting QDrant binary...")
    
    # Create data directory
    data_dir = Path.cwd() / "qdrant_data"
    data_dir.mkdir(exist_ok=True)
    
    cmd = [
        str(binary_path),
        "--storage-path", str(data_dir),
        "--http-port", "6333",
        "--grpc-port", "6334"
    ]
    
    try:
        # Start QDrant in background
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"QDrant started with PID: {process.pid}")
        
        # Save PID for cleanup
        pid_file = Path.cwd() / "qdrant.pid"
        with open(pid_file, "w") as f:
            f.write(str(process.pid))
        
        return True
    except Exception as e:
        print(f"Failed to start QDrant binary: {e}")
        return False




def stop_qdrant():
    """Stop QDrant"""
    print("Stopping QDrant...")
    
    # Try Docker first
    try:
        subprocess.run(["docker", "stop", "myna-qdrant"], check=True, capture_output=True)
        subprocess.run(["docker", "rm", "myna-qdrant"], capture_output=True)
        print("✓ Stopped QDrant Docker container")
        return True
    except subprocess.CalledProcessError:
        pass
    
    # Try PID file
    pid_file = Path.cwd() / "qdrant.pid"
    if pid_file.exists():
        try:
            with open(pid_file, "r") as f:
                pid = int(f.read().strip())
            
            os.kill(pid, 15)  # SIGTERM
            pid_file.unlink()
            print(f"✓ Stopped QDrant process (PID: {pid})")
            return True
        except (OSError, ValueError):
            pass
    
    print("No QDrant instance found to stop")
    return False


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "stop":
        stop_qdrant()
        return
    
    print("Setting up QDrant for Myna...")
    
    # Try Docker first
    if check_docker():
        if start_qdrant_docker():
            if wait_for_qdrant():
                print("\n✓ QDrant is running on http://localhost:6333")
                print("Use 'python setup_qdrant.py stop' to stop")
                return
    
    # Fallback to binary
    print("Docker not available, trying binary download...")
    binary_path = download_qdrant_binary()
    if binary_path:
        if start_qdrant_binary(binary_path):
            if wait_for_qdrant():
                print("\n✓ QDrant is running on http://localhost:6333")
                print("Use 'python setup_qdrant.py stop' to stop")
                return
    
    print("\n✗ Failed to start QDrant")
    print("You can:")
    print("1. Install Docker and try again")
    print("2. Install QDrant manually: https://qdrant.tech/documentation/quick-start/")
    print("3. Use QDrant Cloud: https://cloud.qdrant.io/")


if __name__ == "__main__":
    main()