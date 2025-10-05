"""
Test script to verify the best model is working properly with synthetic test data
"""
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import io
from object_detection_model import HomeObjectDetectionModel, draw_bounding_boxes, HOME_OBJECTS, NUM_CLASSES

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HomeObjectDetectionModel(num_classes=NUM_CLASSES)

# Try loading the best model
model_paths = ["best_detection_model.pth", "best_detection_model_epoch_10.pth", "home_objects_detection_model.pth"]
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
    print("Could not load any model!")
else:
    model.to(device)
    model.eval()

    # Test transformation
    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create a synthetic image with obvious objects to detect
    test_img = Image.new('RGB', (416, 416), color='white')
    draw = ImageDraw.Draw(test_img)

    # Draw multiple shapes that could be recognizable as objects
    draw.rectangle([50, 50, 150, 150], outline='red', width=5)
    draw.text([60, 60], 'chair', fill='red')
    
    draw.ellipse([200, 200, 300, 300], outline='blue', width=5)
    draw.text([210, 210], 'sofa', fill='blue')

    # Apply transformation
    input_tensor = transform(test_img).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        detections = model.predict(input_tensor, conf_thresh=0.1)  # Very low confidence threshold

    print(f"Detections found: {len(detections)}")
    print(f"First batch detections count: {len(detections[0])}")
    if detections[0]:
        for i, det in enumerate(detections[0]):
            x_center, y_center, width, height, conf, cls_id = det
            print(f"Detection {i}: Class={HOME_OBJECTS[int(cls_id)]}, Conf={conf:.3f}, Box=[{x_center:.3f}, {y_center:.3f}, {width:.3f}, {height:.3f}]")
    else:
        print("No detections above the confidence threshold")

    # Try drawing bounding boxes (even if empty)
    result_img = draw_bounding_boxes(test_img, detections[0])
    print("Bounding boxes drawn (or attempted)")
    result_img.save("test_best_model_output.jpg", "JPEG")
    print("Test output saved as test_best_model_output.jpg")