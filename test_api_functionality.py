"""
Test script to simulate the API endpoint functionality
"""
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import io
from object_detection_model import HomeObjectDetectionModel, draw_bounding_boxes, HOME_OBJECTS, NUM_CLASSES

def test_api_endpoint_functionality():
    """Simulate the API endpoint's object detection functionality"""
    
    # Load the model (using same logic as API server)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HomeObjectDetectionModel(num_classes=NUM_CLASSES)

    # Try to load the trained model with the same logic as the API server
    model_paths = [
        "best_detection_model.pth",           # Final best model with metadata
        "home_objects_detection_model.pth",   # Final model state dict
        "home_objects_cnn.pth",               # Expected API model
        "best_detection_model_epoch_10.pth",  # Last epoch
        "best_detection_model_epoch_9.pth",   # Previous epochs as fallbacks
        "best_detection_model_epoch_8.pth",
        "best_detection_model_epoch_7.pth",
        "best_detection_model_epoch_6.pth",
        "best_detection_model_epoch_5.pth",
        "best_detection_model_epoch_4.pth",
        "best_detection_model_epoch_3.pth",
        "best_detection_model_epoch_2.pth",
        "best_detection_model_epoch_1.pth",   # First epoch as fallback
    ]
    
    model_loaded = False

    for model_path in model_paths:
        if torch.load.__doc__ and model_path:  # Just to check path exists
            try:
                import os
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=device)
                    
                    # Check if it's a checkpoint dictionary with 'model_state_dict' key
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                        # Load anchors if available in the checkpoint
                        if 'anchors' in checkpoint:
                            model.anchors = checkpoint['anchors']
                    else:
                        # If it's just the state dict directly, load it normally
                        model.load_state_dict(checkpoint)
                    
                    print(f"Model loaded from {model_path}")
                    model_loaded = True
                    break
            except Exception as e:
                print(f"Could not load model from {model_path}: {e}")

    if not model_loaded:
        print("No model file found. Using untrained model.")
        return

    model.to(device)
    model.eval()

    # Same transformation as in API server
    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create a test image similar to what would be uploaded
    test_img = Image.new('RGB', (416, 416), color='lightblue')
    draw = ImageDraw.Draw(test_img)
    
    # Add some objects that might be detectable
    draw.rectangle([100, 100, 200, 200], outline='red', width=3)
    draw.ellipse([250, 250, 350, 350], outline='green', width=3)
    draw.polygon([(50, 300), (150, 350), (50, 400)], outline='blue', width=3)
    
    # Apply the same transformation as in the API server
    input_tensor = transform(test_img).unsqueeze(0).to(device)
    
    # Run inference (testing with different confidence thresholds)
    print("\nTesting detection with various confidence thresholds:")
    for conf_thresh in [0.1, 0.3, 0.5, 0.7]:
        with torch.no_grad():
            detections = model.predict(input_tensor, conf_thresh=conf_thresh)
        
        print(f"Confidence threshold {conf_thresh}: {len(detections[0])} detections found")
        if detections[0]:
            for i, det in enumerate(detections[0]):
                x_center, y_center, width, height, conf, cls_id = det
                print(f"  Detection {i}: {HOME_OBJECTS[int(cls_id)]} ({conf:.3f}) at center ({x_center:.3f}, {y_center:.3f})")
    
    # Test with lowest confidence threshold to see if we get any results
    with torch.no_grad():
        detections = model.predict(input_tensor, conf_thresh=0.1)
    
    print(f"\nUsing 0.1 threshold: {len(detections[0])} detections")
    
    # Draw bounding boxes (this will work whether or not we found detections)
    result_img = draw_bounding_boxes(test_img, detections[0])
    result_img.save("api_test_output.jpg", "JPEG")
    print("Test output saved as api_test_output.jpg")
    
    # Prepare detection results like the API would
    detection_results = []
    for det in detections[0]:  # detections[0] because batch size is 1
        x_center, y_center, width, height, conf, cls_id = det
        x1 = int((x_center - width/2) * test_img.width)
        y1 = int((y_center - height/2) * test_img.height)
        x2 = int((x_center + width/2) * test_img.width)
        y2 = int((y_center + height/2) * test_img.height)
        
        detection_results.append({
            "class": HOME_OBJECTS[int(cls_id)],
            "confidence": float(conf),
            "bbox": [x1, y1, x2, y2]  # [x_min, y_min, x_max, y_max]
        })
    
    print(f"\nDetection results format (as API would return):")
    for i, result in enumerate(detection_results):
        print(f"  Object {i+1}: {result['class']} ({result['confidence']:.3f}) at {result['bbox']}")

if __name__ == "__main__":
    test_api_endpoint_functionality()