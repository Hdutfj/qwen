"""
Test Script for Home Objects Detection System
This script tests the integration of all system components.
"""
import requests
import os
from PIL import Image, ImageDraw
import io
import base64

def create_test_image():
    """Create a simple test image with a recognizable shape"""
    # Create a 416x416 test image
    img = Image.new('RGB', (416, 416), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a rectangle (might be detected as a chair, table, etc.)
    draw.rectangle([100, 100, 300, 300], outline='red', width=5)
    
    # Draw a circle (might be detected as a clock, vase, etc.)
    draw.ellipse([50, 50, 150, 150], outline='blue', width=5)
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG', quality=90)
    img_bytes.seek(0)
    
    return img_bytes

def test_api_connection():
    """Test if the API server is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False
    except requests.exceptions.Timeout:
        return False

def test_detection_api():
    """Test the detection API endpoint"""
    try:
        # Create test image
        test_img = create_test_image()
        
        # Prepare form data
        files = {'image': ('test_image.jpg', test_img, 'image/jpeg')}
        data = {'confidence_threshold': '0.3'}
        
        # Send request to detection endpoint
        response = requests.post(
            "http://localhost:8000/detect",
            files=files,
            data=data,
            timeout=30
        )
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing detection API: {e}")
        return False

def test_multiple_detection_api():
    """Test the multiple detection API endpoint"""
    try:
        # Create two test images
        test_img1 = create_test_image()
        test_img2 = create_test_image()
        
        # Prepare form data
        files = [
            ('images', ('test_image1.jpg', test_img1, 'image/jpeg')),
            ('images', ('test_image2.jpg', test_img2, 'image/jpeg'))
        ]
        data = {'confidence_threshold': '0.3'}
        
        # Send request to multiple detection endpoint
        response = requests.post(
            "http://localhost:8000/detect-multiple",
            files=files,
            data=data,
            timeout=60
        )
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing multiple detection API: {e}")
        return False

def test_classes_endpoint():
    """Test the classes endpoint"""
    try:
        response = requests.get("http://localhost:8000/classes", timeout=10)
        return response.status_code == 200 and 'classes' in response.json()
    except Exception as e:
        print(f"Error testing classes endpoint: {e}")
        return False

def run_all_tests():
    """Run all integration tests"""
    print("Home Objects Detection System - Integration Tests")
    print("=" * 60)
    
    tests = [
        ("API Connection", test_api_connection),
        ("Detection API", test_detection_api),
        ("Multiple Detection API", test_multiple_detection_api),
        ("Classes Endpoint", test_classes_endpoint),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        try:
            result = test_func()
            results[test_name] = result
            status = "PASS" if result else "FAIL"
            print(f"  {test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            print(f"  {test_name}: ERROR - {e}")
        print()
    
    # Summary
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print("Test Summary:")
    print(f"  Passed: {passed}/{total}")
    print(f"  Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\nAll tests passed! The system is working correctly.")
        return True
    else:
        print("\nSome tests failed. Please check the system configuration.")
        return False

if __name__ == "__main__":
    run_all_tests()