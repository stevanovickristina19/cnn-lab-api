# app/services/predict_service.py
import torch
import torch.nn.functional as F
from app.models.models import get_model
from torchvision import transforms
from PIL import Image
import io, base64, os, json

device = "cuda" if torch.cuda.is_available() else "cpu"


DEFAULT_CLASS_NAMES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

def load_model_config(model_name):
    """Load the saved config (JSON) so num_classes and labels match training."""
    config_path = f"saved_models/{model_name}_config.json"
    if not os.path.exists(config_path):
        print(f"No config found for {model_name}, using default num_classes=10.")
        return {"num_classes": 10, "class_names": DEFAULT_CLASS_NAMES}

    with open(config_path, "r") as f:
        config = json.load(f)

    # Fallback defaults
    config.setdefault("num_classes", 10)
    config.setdefault("class_names", DEFAULT_CLASS_NAMES)
    return config


def predict(image_base64, model_name):
    # Decode image from base64
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Load config for correct num_classes and labels
    config = load_model_config(model_name)
    num_classes = config["num_classes"]
    class_names = config["class_names"]

    # Build model with correct num_classes
    model = get_model(model_name, num_classes=num_classes).to(device)

    # Load trained weights
    model_path = f"saved_models/{model_name}.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f" Model weights not found: {model_path}")

    print(f" Loading {model_name} with {num_classes} classes...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Apply transforms (same as during training)
    input_size = getattr(config, "input_size", (128, 128))

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(F.softmax(output, dim=1), 1)
        label = class_names[predicted.item()] if predicted.item() < len(class_names) else str(predicted.item())

    return label
