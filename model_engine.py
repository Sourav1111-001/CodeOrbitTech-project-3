"""
Model Engine for VisionLens AI
-------------------------------
Supports multiple pretrained PyTorch vision models (MobileNetV2, ResNet18, ResNet50, EfficientNet-B0),
preprocessing, top-k inference, and Grad-CAM explainable AI visualization.
"""

import time
import io
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import (
    MobileNet_V2_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    EfficientNet_B0_Weights,
)
from PIL import Image
import numpy as np
import cv2

# Global cache for loaded models
_LOADED_MODELS = {}

MODEL_INFO = {
    "mobilenet_v2": {
        "name": "MobileNetV2",
        "description": "Ultra-lightweight CNN optimized for edge devices & mobile inference.",
        "params": "3.5M",
        "input_size": "224x224",
        "speed": "Fastest (~15-30ms CPU)",
        "accuracy": "71.9% Top-1",
        "weights": MobileNet_V2_Weights.DEFAULT,
        "loader": models.mobilenet_v2,
        "target_layer_name": "features"
    },
    "resnet18": {
        "name": "ResNet-18",
        "description": "18-layer Deep Residual Network with skip connections to avoid vanishing gradients.",
        "params": "11.7M",
        "input_size": "224x224",
        "speed": "Fast (~25-45ms CPU)",
        "accuracy": "69.8% Top-1",
        "weights": ResNet18_Weights.DEFAULT,
        "loader": models.resnet18,
        "target_layer_name": "layer4"
    },
    "resnet50": {
        "name": "ResNet-50",
        "description": "Industry benchmark 50-layer deep residual architecture for high accuracy.",
        "params": "25.6M",
        "input_size": "224x224",
        "speed": "Balanced (~60-110ms CPU)",
        "accuracy": "76.1% Top-1",
        "weights": ResNet50_Weights.DEFAULT,
        "loader": models.resnet50,
        "target_layer_name": "layer4"
    },
    "efficientnet_b0": {
        "name": "EfficientNet-B0",
        "description": "Compound scaled neural network balancing depth, width, and resolution.",
        "params": "5.3M",
        "input_size": "224x224",
        "speed": "Very Fast (~30-55ms CPU)",
        "accuracy": "77.1% Top-1",
        "weights": EfficientNet_B0_Weights.DEFAULT,
        "loader": models.efficientnet_b0,
        "target_layer_name": "features"
    }
}

def get_model(model_id: str = "mobilenet_v2"):
    """Loads and caches the requested pretrained model."""
    if model_id not in MODEL_INFO:
        model_id = "mobilenet_v2"
        
    if model_id not in _LOADED_MODELS:
        info = MODEL_INFO[model_id]
        weights = info["weights"]
        model = info["loader"](weights=weights)
        model.eval()
        _LOADED_MODELS[model_id] = {
            "model": model,
            "preprocess": weights.transforms(),
            "categories": weights.meta["categories"],
            "info": info
        }
    return _LOADED_MODELS[model_id]

class GradCAM:
    """Computes Grad-CAM (Gradient-weighted Class Activation Mapping) for visual explainability."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor, target_class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_class_idx is None:
            target_class_idx = output.argmax(dim=1).item()

        score = output[0, target_class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients[0] # (C, H, W)
        activations = self.activations[0] # (C, H, W)

        # Global average pooling on gradients -> weights
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        
        # Normalize to 0-1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy(), target_class_idx

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

def classify_and_explain(pil_img: Image.Image, model_id: str = "mobilenet_v2", top_k: int = 5, generate_heatmap: bool = True):
    """
    Runs inference on an image and optionally generates a Grad-CAM heatmap.
    """
    model_data = get_model(model_id)
    model = model_data["model"]
    preprocess = model_data["preprocess"]
    categories = model_data["categories"]
    target_layer_name = model_data["info"]["target_layer_name"]
    target_layer = getattr(model, target_layer_name)

    # Preprocess image
    rgb_img = pil_img.convert("RGB")
    tensor = preprocess(rgb_img).unsqueeze(0)

    # Measure latency
    start_time = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Extract Top-K
    top_preds = torch.topk(probabilities, top_k)
    results = []
    for prob, idx in zip(top_preds.values, top_preds.indices):
        idx_val = idx.item()
        results.append({
            "class_id": idx_val,
            "label": categories[idx_val].replace("_", " ").title(),
            "confidence": round(prob.item() * 100, 2),
            "raw_prob": float(prob.item())
        })

    heatmap_base64 = None
    if generate_heatmap:
        try:
            tensor_req_grad = tensor.clone().requires_grad_(True)
            grad_cam = GradCAM(model, target_layer)
            cam_map, _ = grad_cam.generate(tensor_req_grad, target_class_idx=results[0]["class_id"])
            grad_cam.remove_hooks()

            # Resize CAM map to original image size
            orig_w, orig_h = rgb_img.size
            cam_resized = cv2.resize(cam_map, (orig_w, orig_h))
            cam_heatmap = np.uint8(255 * cam_resized)
            cam_colored = cv2.applyColorMap(cam_heatmap, cv2.COLORMAP_JET)
            cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)

            # Blend with original image
            orig_np = np.array(rgb_img)
            overlay = np.uint8(0.55 * orig_np + 0.45 * cam_colored)

            # Convert to base64
            overlay_pil = Image.fromarray(overlay)
            buf = io.BytesIO()
            overlay_pil.save(buf, format="JPEG", quality=85)
            import base64
            heatmap_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"Heatmap generation error: {e}")
            heatmap_base64 = None

    return {
        "model_id": model_id,
        "model_info": model_data["info"],
        "latency_ms": latency_ms,
        "predictions": results,
        "heatmap_base64": heatmap_base64
    }
