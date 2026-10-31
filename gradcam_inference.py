import io
import os
import base64
import logging
from typing import Union, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

logger = logging.getLogger("neuroscan.inference")

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Determine optimal available device (CUDA for AWS GPU instances, MPS for Mac, CPU fallback)
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)

# Standard ImageNet transform matching training pipeline
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

_MODEL: Optional[nn.Module] = None
_MODEL_PATH: str = os.getenv("MODEL_PATH", "best_model.pth")


def load_model(weights_path: str = _MODEL_PATH) -> nn.Module:
    """
    Initializes ResNet18 and loads trained weights once into memory.
    Keeps model in evaluation mode on the target device.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    logger.info(f"Initializing ResNet18 architecture for {len(CLASSES)} classes on {DEVICE}...")
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))

    if not os.path.exists(weights_path):
        remote_url = os.getenv(
            "MODEL_URL",
            "https://huggingface.co/Ahmed-Selem/Shifaa-Brain-Tumor-ResNet18/resolve/main/braintumor_model_weights.pth"
        )
        logger.warning(f"Model file '{weights_path}' not found locally. Downloading from {remote_url}...")
        try:
            import urllib.request
            urllib.request.urlretrieve(remote_url, weights_path)
            logger.info(f"Successfully downloaded model weights to {weights_path}.")
        except Exception as dl_err:
            logger.error(f"Failed to download model weights: {dl_err}")

    if os.path.exists(weights_path):
        logger.info(f"Loading weights from {weights_path}...")
        try:
            state_dict = torch.load(weights_path, weights_only=True, map_location=DEVICE)
            model.load_state_dict(state_dict)
            logger.info("Successfully loaded trained model weights.")
        except Exception as e:
            logger.error(f"Error loading state dict from {weights_path}: {e}")
            raise
    else:
        logger.warning(
            f"Model weights file '{weights_path}' could not be loaded. Running with base weights."
        )

    model.to(DEVICE)
    model.eval()
    _MODEL = model
    return _MODEL


def is_model_ready() -> bool:
    """Returns True if the model is initialized and weights are ready."""
    return _MODEL is not None


def get_model() -> nn.Module:
    """Returns the cached model instance, loading it if not yet loaded."""
    if _MODEL is None:
        return load_model()
    return _MODEL


def predict_and_explain(
    image_input: Union[str, bytes, io.BytesIO, Image.Image]
) -> Dict[str, Any]:
    """
    Predicts the brain tumor class of an image and generates a Grad-CAM heatmap.

    Args:
        image_input: File path (str), raw image bytes, BytesIO buffer, or PIL Image.

    Returns:
        Dict containing:
          - predicted_class: str
          - confidence: float
          - all_probabilities: Dict[str, float]
          - heatmap_base64: str (base64-encoded JPEG)
    """
    model = get_model()

    # 1. Parse image into PIL RGB Image
    if isinstance(image_input, Image.Image):
        original_img_pil = image_input.convert("RGB")
    elif isinstance(image_input, (bytes, bytearray)):
        original_img_pil = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, io.BytesIO):
        original_img_pil = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found at path: {image_input}")
        original_img_pil = Image.open(image_input).convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # 2. Preprocess
    original_img_resized = original_img_pil.resize((224, 224))
    rgb_img = np.float32(original_img_resized) / 255.0

    input_tensor = TRANSFORM(original_img_pil).unsqueeze(0).to(DEVICE)

    # 3. Model Inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output[0], dim=0)

        all_probabilities = {
            CLASSES[i]: float(probabilities[i].item()) for i in range(len(CLASSES))
        }

        max_prob, top_catid = torch.max(probabilities, 0)
        predicted_class_idx = top_catid.item()
        predicted_class = CLASSES[predicted_class_idx]
        confidence = float(max_prob.item())

    # 4. Generate Grad-CAM Heatmap
    # Target last conv layer in ResNet18 (layer4[-1])
    target_layers = [model.layer4[-1]]
    targets = [ClassifierOutputTarget(predicted_class_idx)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    heatmap_img = Image.fromarray(visualization)

    # 5. Convert heatmap to base64
    buffered = io.BytesIO()
    heatmap_img.save(buffered, format="JPEG", quality=90)
    heatmap_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": all_probabilities,
        "heatmap_base64": heatmap_base64,
    }


# Pre-warm model on module import so first request has zero cold-start delay
try:
    load_model()
except Exception as _e:
    logger.warning(f"Deferred model loading: {_e}")


if __name__ == "__main__":
    import sys
    print(f"Testing gradcam_inference on {DEVICE}...")
    sample_path = os.path.join("sample_images", "glioma.jpg")
    if os.path.exists(sample_path):
        res = predict_and_explain(sample_path)
        print(f"Prediction: {res['predicted_class']} (Confidence: {res['confidence'] * 100:.2f}%)")
        print("Probabilities:", res["all_probabilities"])
        print(f"Heatmap base64 preview: {res['heatmap_base64'][:50]}... (length: {len(res['heatmap_base64'])})")
    else:
        print(f"Sample image {sample_path} not found.")
