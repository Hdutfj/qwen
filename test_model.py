"""
Test script to verify model is working properly
"""
import torch
from PIL import Image
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

# Load a sample image (using a blank image for testing)
test_img = Image.new('RGB', (416, 416), color='blue')

# Apply transformation
input_tensor = transform(test_img).unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    detections = model.predict(input_tensor, conf_thresh=0.5)

print(f"Detections found: {len(detections)}")
print(f"First batch detections: {detections[0]}")

# Try drawing bounding boxes
result_img = draw_bounding_boxes(test_img, detections[0])
print("Bounding boxes drawn successfully")
result_img.save("test_output.jpg", "JPEG")
print("Test output saved as test_output.jpg")