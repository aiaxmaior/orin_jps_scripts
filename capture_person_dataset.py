#!/usr/bin/env python3
"""
Person Detection Dataset Collector - Python wrapper
Uses the C implementation for DeepStream person capture.

Compile the C version first:
    make deepstream_capture_person

Or manually:
    gcc -o deepstream_capture_person deepstream_capture_person.c \
        $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-video-1.0) \
        -I/opt/nvidia/deepstream/deepstream/sources/includes \
        -L/opt/nvidia/deepstream/deepstream/lib -lnvdsgst_meta -lnvds_meta \
        -lnvbufsurface -lnvbufsurftransform -ljpeg \
        -Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib
"""

import subprocess
import sys
import os

def main():
    # Check if C binary exists
    binary = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deepstream_capture_person')

    if not os.path.exists(binary):
        print("=" * 60)
        print("ERROR: C binary not found. Please compile first:")
        print("=" * 60)
        print()
        print("  make deepstream_capture_person")
        print()
        print("Or manually:")
        print("  gcc -o deepstream_capture_person deepstream_capture_person.c \\")
        print("      $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-video-1.0) \\")
        print("      -I/opt/nvidia/deepstream/deepstream/sources/includes \\")
        print("      -L/opt/nvidia/deepstream/deepstream/lib -lnvdsgst_meta -lnvds_meta \\")
        print("      -lnvbufsurface -lnvbufsurftransform -ljpeg \\")
        print("      -Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib")
        print("=" * 60)
        return 1

    # Pass all arguments to C binary
    cmd = [binary] + sys.argv[1:]

    try:
        return subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        print("\nStopped")
        return 0


if __name__ == '__main__':
    sys.exit(main())
