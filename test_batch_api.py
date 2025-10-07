import requests
import os
from PIL import Image
import io

# Test batch detection API
def test_batch_detection():
    url = "http://localhost:8000/detect-batch"
    
    # Create some dummy images for testing
    images = []
    for i in range(2):
        # Create a dummy image
        img = Image.new('RGB', (200, 200), color=(255, 255, 255))
        
        # Draw something simple
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 150, 150], outline=(0, 0, 0), width=3)
        draw.ellipse([75, 75, 125, 125], outline=(255, 0, 0), width=3)
        
        # Save to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        images.append(('images', (f'test_image_{i}.jpg', img_bytes, 'image/jpeg')))
    
    # Send request
    try:
        response = requests.post(url, files=images)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_batch_detection()