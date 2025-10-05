"""
Simple test to check import functionality
"""
import sys
print("Python version:", sys.version)

# Test basic imports
try:
    import torch
    print("✓ Torch imported successfully")
    print("Torch version:", torch.__version__)
except Exception as e:
    print("✗ Error importing torch:", e)

try:
    import torchvision
    print("✓ Torchvision imported successfully")
    print("Torchvision version:", torchvision.__version__)
except Exception as e:
    print("✗ Error importing torchvision:", e)

try:
    from PIL import Image
    print("✓ PIL imported successfully")
except Exception as e:
    print("✗ Error importing PIL:", e)

try:
    import numpy
    print("✓ Numpy imported successfully")
except Exception as e:
    print("✗ Error importing numpy:", e)

print("Import test completed.")