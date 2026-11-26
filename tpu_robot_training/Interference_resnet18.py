import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

CHECKPOINT = "/home/robot/Downloads/best_weights_epoch16.pt"
IMAGE = "/home/robot/Desktop/Run_02_Melt_Mix_test_f0027704_t0692.60s.jpg"

# Classes from the original training dataset
CLASSES = {
    0: "Dissolved",
    1: "Undissolved"
}

# Build the same ResNet18 architecture used during training
model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 2)

# Load trained weights
state_dict = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True
)
model.load_state_dict(state_dict)
model.eval()

# Same preprocessing used for validation/test
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load image
image = Image.open(IMAGE).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Run inference
start = time.perf_counter()

with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

end = time.perf_counter()

prediction = CLASSES[predicted_class]
confidence = probabilities[0, predicted_class].item() * 100
inference_time = (end - start) * 1000

print()
print("===================================")
print("RESNET18 INFERENCE TEST")
print("===================================")
print(f"Image: {IMAGE}")
print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.2f}%")
print(f"Inference time: {inference_time:.2f} ms")
print("===================================")