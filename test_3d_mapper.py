"""
Test script for the AI 3D Scene Mapper feature
This script tests the complete pipeline: 2D detection -> depth estimation -> 3D reconstruction -> visualization
"""
import requests
import json
from PIL import Image, ImageDraw
import io
import uuid
import os
from pathlib import Path

def create_test_image():
    """Create a test image with known objects"""
    # Create a test image with recognizable objects
    img = Image.new('RGB', (416, 416), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Draw some objects that our model can recognize
    draw.rectangle([100, 100, 200, 200], fill='red', outline='black', width=3)
    draw.ellipse([250, 150, 350, 250], fill='green', outline='black', width=3)
    draw.rectangle([50, 300, 150, 380], fill='blue', outline='black', width=3)
    
    # Add labels (these won't be detected by the AI, they're just for reference)
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
        draw.text((100, 80), "toilet", fill='black', font=font)
        draw.text((250, 130), "sink", fill='black', font=font)
        draw.text((50, 280), "mirror", fill='black', font=font)
    except:
        draw.text((100, 80), "toilet", fill='black')
        draw.text((250, 130), "sink", fill='black')
        draw.text((50, 280), "mirror", fill='black')
    
    return img


def test_api_connection():
    """Test if the API is running"""
    try:
        response = requests.get("http://localhost:8000/health")
        return response.status_code == 200
    except:
        return False


def test_3d_scene_mapping():
    """Test the complete 3D scene mapping functionality"""
    if not test_api_connection():
        print("❌ API server is not running. Please start the API server first.")
        print("Run: python api_server.py")
        return False
    
    print("✅ API server is running")
    
    # Create a test image
    test_img = create_test_image()
    
    # Save to bytes for upload
    img_byte_arr = io.BytesIO()
    test_img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Prepare the request
    files = {
        'image': (f'test_image_{uuid.uuid4()}.jpg', img_byte_arr, 'image/jpeg')
    }
    
    data = {
        'detection_method': 'openai'  # Using openai method as default
    }
    
    try:
        print("🔍 Sending request to 3D scene mapping API...")
        response = requests.post(
            "http://localhost:8000/3d-scene-map",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 3D scene mapping successful!")
            print(f"   - Input image: {result['filename']}")
            print(f"   - Detection method: {result['detection_source']}")
            print(f"   - Objects detected: {result['detection_count']}")
            print(f"   - Objects in 3D: {result['object_count_3d']}")
            print(f"   - Detected objects: {result['detected_objects']}")
            
            # Check if 3D objects were created
            if result['objects_3d']:
                print("   - 3D objects created:")
                for obj in result['objects_3d']:
                    print(f"     * {obj['class']}: pos={obj['position_3d'][:2]}..., dims={obj['dimensions_3d'][:2]}..., conf={obj['confidence']:.2f}")
            
            # Check for 3D visualization URL
            if result.get('visualization_3d_url'):
                print(f"   - 3D Visualization URL: {result['visualization_3d_url']}")
                print(f"   - Scene data URL: {result['scene_data_url']}")
                print("\n🎉 You can view the 3D visualization at:")
                print(f"   http://localhost:8000{result['visualization_3d_url']}")
            
            return True
        else:
            print(f"❌ API request failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during 3D scene mapping test: {str(e)}")
        return False


def test_basic_detection():
    """Test basic object detection to ensure the system is working"""
    if not test_api_connection():
        print("❌ API server is not running. Please start the API server first.")
        print("Run: python api_server.py")
        return False
    
    # Create a test image
    test_img = create_test_image()
    
    # Save to bytes for upload
    img_byte_arr = io.BytesIO()
    test_img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Prepare the request
    files = {
        'image': (f'test_image_{uuid.uuid4()}.jpg', img_byte_arr, 'image/jpeg')
    }
    
    data = {
        'detection_method': 'openai'
    }
    
    try:
        print("🔍 Sending request to basic detection API...")
        response = requests.post(
            "http://localhost:8000/detect",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Basic detection successful!")
            print(f"   - Objects detected: {result['detection_count']}")
            print(f"   - Detected objects: {result['detected_objects']}")
            return True
        else:
            print(f"❌ Basic detection failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during basic detection test: {str(e)}")
        return False


def main():
    """Main test function"""
    print("="*60)
    print("🧪 Testing AI 3D Scene Mapper Feature")
    print("="*60)
    
    # Test basic functionality first
    print("\n🔍 Testing basic object detection...")
    basic_success = test_basic_detection()
    
    if basic_success:
        print("\n🔍 Testing 3D scene mapping functionality...")
        mapping_success = test_3d_scene_mapping()
        
        print("\n" + "="*60)
        if mapping_success:
            print("🎉 All tests passed! The AI 3D Scene Mapper is working correctly.")
            print("\nNext steps:")
            print("1. Try uploading your own images to the API")
            print("2. Access the 3D visualization via the returned URL")
            print("3. Explore the interactive 3D scene in your browser")
        else:
            print("❌ 3D scene mapping test failed.")
    else:
        print("❌ Basic detection failed. Please check your API setup.")
    
    print("="*60)


if __name__ == "__main__":
    main()