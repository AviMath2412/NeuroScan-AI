import io
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Dummy classes as requested - Updated to match dataset folder names
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def predict_and_explain(image_path):
    """
    Predicts the class of an image and generates a GradCAM heatmap.
    Returns a dictionary containing predicted_class, confidence, all_probabilities, and heatmap_base64.
    """
    # 1. Load the ResNet18 architecture
    model = models.resnet18(weights=None) # We don't need pretrained weights as we load our own
    
    # Replace the last fully connected layer to match our training (4 classes)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    
    # Load our trained weights
    model_path = 'best_model.pth'
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    except Exception as e:
        print(f"Warning: Could not load trained weights from {model_path}. Did you run train.py? Error: {e}")
        # Proceed with random weights just so the script doesn't completely crash for testing
    
    model.eval()

    # Define the target layer for GradCAM (last convolutional layer in ResNet18)
    target_layers = [model.layer4[-1]]

    # 2. Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load original image for visualization
    try:
        original_img_pil = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    original_img_resized = original_img_pil.resize((224, 224))
    # Convert to float [0, 1] for show_cam_on_image
    rgb_img = np.float32(original_img_resized) / 255.0

    input_tensor = transform(original_img_pil).unsqueeze(0) # Add batch dimension

    # 3. Run inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output[0], dim=0)
        
        # Get all probabilities mapped to class names
        all_probabilities = {CLASSES[i]: float(probabilities[i].item()) for i in range(len(CLASSES))}
        
        # Get the highest probability
        max_prob, top_catid = torch.max(probabilities, 0)
        
        # The model directly predicts our 4 classes now
        predicted_class_idx = top_catid.item()
        predicted_class = CLASSES[predicted_class_idx]
        confidence = float(max_prob.item())

    # 4. Generate GradCAM heatmap
    # Initialize CAM object
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Target the predicted class
    targets = [ClassifierOutputTarget(top_catid.item())]
    
    # Generate the CAM mask
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    
    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    
    # Overlay the heatmap on the original image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # Convert back to PIL Image
    heatmap_img = Image.fromarray(visualization)

    # 5. Convert heatmap to base64 string
    buffered = io.BytesIO()
    heatmap_img.save(buffered, format="JPEG")
    heatmap_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": all_probabilities,
        "heatmap_base64": heatmap_base64
    }

def get_model_evaluation_metrics():
    """
    Computes and returns the overall accuracy and false positive breakdown
    across all test classes and clinical screening.
    """
    from evaluate_model import evaluate_model
    return evaluate_model()

if __name__ == "__main__":
    import os
    
    # Mock execution block
    # Use a real image from the testing set instead of a dummy image
    test_image_dir = os.path.join("data", "Testing", "glioma")
    test_images = [f for f in os.listdir(test_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if test_images:
        test_image_path = os.path.join(test_image_dir, test_images[0])
        print(f"Running inference on real test image: {test_image_path}...")
        result = predict_and_explain(test_image_path)
    else:
        print(f"No test images found in {test_image_dir}. Please check your dataset.")
        result = None
    
    if result:
        print(f"Prediction: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("All Probabilities:")
        for cls_name, prob in result['all_probabilities'].items():
            print(f"  {cls_name}: {prob:.4f}")
        print(f"Heatmap Base64 (preview): {result['heatmap_base64'][:60]}... (length: {len(result['heatmap_base64'])})")
