"""
Test script to verify model is working properly with synthetic test data
"""
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import io
from object_detection_model import HomeObjectDetectionModel, draw_bounding_boxes, HOME_OBJECTS, NUM_CLASSES

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HomeObjectDetectionModel(num_classes=NUM_CLASSES)

# Load the trained model
model_path = "home_objects_detection_model.pth"
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
except Exception as e:
    print(f"Could not load model from {model_path}: {e}")

model.to(device)
model.eval()

# Test transformation
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create a synthetic image with objects to detect
test_img = Image.new('RGB', (416, 416), color='lightblue')
draw = ImageDraw.Draw(test_img)

# Draw a simple shape that might be recognized as an object
draw.rectangle([100, 100, 200, 200], outline='red', width=3)
draw.text([110, 110], 'chair', fill='red')

# Apply transformation
input_tensor = transform(test_img).unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    detections = model.predict(input_tensor, conf_thresh=0.3)  # Lowered confidence threshold

print(f"Detections found: {len(detections)}")
print(f"First batch detections count: {len(detections[0])}")
if detections[0]:
    for i, det in enumerate(detections[0]):
        x_center, y_center, width, height, conf, cls_id = det
        print(f"Detection {i}: Class={HOME_OBJECTS[int(cls_id)]}, Conf={conf:.3f}, Box=[{x_center:.3f}, {y_center:.3f}, {width:.3f}, {height:.3f}]")

# Try drawing bounding boxes
result_img = draw_bounding_boxes(test_img, detections[0])
print("Bounding boxes drawn successfully")
result_img.save("test_output_with_objects.jpg", "JPEG")
print("Test output saved as test_output_with_objects.jpg")