#!/usr/bin/env python3
# DeepStream proximity app - subprocess wrapper

import subprocess
import sys

def run_deepstream_with_proximity(config_file):
    """Launch deepstream-app as subprocess"""
    try:
        cmd = ['deepstream-app', '-c', config_file]
        print(f"Running: {' '.join(cmd)}")
        
        proc = subprocess.run(cmd, check=True)
        return proc.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"DeepStream failed with code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nStopped by user")
        return 0

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: ./deepstream_proximity_app.py <config_file>")
        print("Example: ./deepstream_proximity_app.py deepstream_single_csi.txt")
        sys.exit(1)
    
    config = sys.argv[1]
    exit_code = run_deepstream_with_proximity(config)
    sys.exit(exit_code)
