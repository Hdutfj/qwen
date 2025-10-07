#!/usr/bin/env python3
"""
Test script to check imports and basic functionality
"""

print("Testing imports...")

try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"✗ PyTorch import failed: {e}")

try:
    from PIL import Image
    print("✓ PIL (Pillow) imported successfully")
except Exception as e:
    print(f"✗ PIL import failed: {e}")

try:
    import numpy as np
    print("✓ NumPy imported successfully")
except Exception as e:
    print(f"✗ NumPy import failed: {e}")

try:
    import matplotlib
    print("✓ Matplotlib imported successfully")
except Exception as e:
    print(f"✗ Matplotlib import failed: {e}")

try:
    import cv2
    print("✓ OpenCV imported successfully")
except Exception as e:
    print(f"✗ OpenCV import failed: {e}")

try:
    from transformers import pipeline, AutoFeatureExtractor
    print("✓ Transformers imported successfully")
except Exception as e:
    print(f"! Transformers import failed (this is expected if not installed): {e}")

try:
    import fastapi
    print("✓ FastAPI imported successfully")
except Exception as e:
    print(f"✗ FastAPI import failed: {e}")

try:
    import requests
    print("✓ Requests imported successfully")
except Exception as e:
    print(f"✗ Requests import failed: {e}")

try:
    from openai import OpenAI
    print("✓ OpenAI imported successfully")
except Exception as e:
    print(f"! OpenAI import failed (this is OK if not installed): {e}")

print("\nTesting basic API functionality without transformers...")
try:
    # Import just the essential parts of depth_3d_mapper
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from depth_3d_mapper import DepthEstimator, PointCloudGenerator, Object3DAnchorer, Scene3DVisualizer, AI3DSceneMapper
    print("✓ Depth mapper classes imported successfully")
    
    # Test creating an instance (this should use the fallback dummy implementation)
    estimator = DepthEstimator()
    print("✓ DepthEstimator created successfully")
    
    mapper = AI3DSceneMapper()
    print("✓ AI3DSceneMapper created successfully")
    
except Exception as e:
    print(f"✗ Depth mapper test failed: {e}")
    import traceback
    traceback.print_exc()

print("Import testing completed.")