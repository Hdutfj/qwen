"""
Debug script to test the draw_bounding_boxes function and model prediction
"""
import torch
from PIL import Image, ImageDraw
import io
import torchvision.transforms as transforms
from object_detection_model import HomeObjectDetectionModel, draw_bounding_boxes, HOME_OBJECTS

def test_model_loading():
    """Test if model loads correctly"""
    print("Testing model loading...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = HomeObjectDetectionModel()
    model_paths = ["home_objects_detection_model.pth", "home_objects_cnn.pth", "best_detection_model.pth"]
    model_loaded = False
    
    for model_path in model_paths:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model loaded from {model_path}")
            model_loaded = True
            break
        except Exception as e:
            print(f"Could not load model from {model_path}: {e}")
    
    if not model_loaded:
        print("No model file found")
        return None
        
    model.to(device)
    model.eval()
    return model

def create_test_image():
    """Create a simple test image similar to the one used in tests"""
    # Create a 416x416 test image
    img = Image.new('RGB', (416, 416), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a rectangle (might be detected as a chair, table, etc.)
    draw.rectangle([100, 100, 300, 300], outline='red', width=5)
    
    # Draw a circle (might be detected as a clock, vase, etc.)
    draw.ellipse([50, 50, 150, 150], outline='blue', width=5)
    
    return img

def test_prediction_and_drawing():
    """Test prediction and drawing functions"""
    print("\nTesting prediction and drawing functions...")
    
    model = test_model_loading()
    if model is None:
        print("Cannot proceed without a loaded model")
        return
    
    # Create test image
    test_img = create_test_image()
    print(f"Created test image: {test_img.size}, mode: {test_img.mode}")
    
    # Transform for model
    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(test_img).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Run prediction
    with torch.no_grad():
        detections = model.predict(input_tensor, conf_thresh=0.3)
    
    print(f"Number of detections: {len(detections)}")
    if detections and len(detections[0]) > 0:
        print(f"First batch detections: {detections[0]}")
        for i, det in enumerate(detections[0]):
            x_center, y_center, width, height, conf, cls_id = det
            print(f"  Detection {i}: class={HOME_OBJECTS[int(cls_id)]}, conf={conf:.3f}")
    else:
        print("No detections found or confidence too low")
    
    # Draw bounding boxes
    print("\nDrawing bounding boxes...")
    result_img = draw_bounding_boxes(test_img, detections[0])
    
    # Save the result to see if it worked
    result_img.save("debug_result.jpg", "JPEG")
    print("Saved result image as debug_result.jpg")
    
    return result_img

if __name__ == "__main__":
    test_prediction_and_drawing()