"""
Simple Image Classification using a Pretrained MobileNetV2 Model (PyTorch)
--------------------------------------------------------------------------
This script loads a lightweight pretrained model (MobileNetV2),
preprocesses sample images, and predicts their labels with confidence scores.
"""

import os
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image

def main():
    print("=" * 60)
    print(" Pretrained Model Image Classification (MobileNetV2) ")
    print("=" * 60)

    # 1. Load Pretrained Model & Preprocessing Pipeline
    print("\n[1] Loading pretrained MobileNetV2 model...")
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.eval()  # Set model to evaluation mode (inference only)

    preprocess = weights.transforms()
    categories = weights.meta["categories"]
    print("    Model successfully loaded with ImageNet weights!\n")

    # 2. Select Sample Images
    images_dir = "saved_images"
    sample_files = [
        "Dog_Golden_20261009_000028.jpg",
        "Dog_Hound_20261009_000029.jpg",
        "Cat_20261009_000032.jpg"
    ]

    # Verify if sample images exist
    images_to_test = [
        os.path.join(images_dir, f) for f in sample_files
        if os.path.exists(os.path.join(images_dir, f))
    ]

    if not images_to_test:
        print("[!] No sample images found in 'saved_images'. Please place image files there.")
        return

    # 3. Classify Each Sample Image
    print(f"[2] Classifying {len(images_to_test)} sample images:\n")
    
    for idx, img_path in enumerate(images_to_test, 1):
        print(f"--- Sample Image {idx}: {os.path.basename(img_path)} ---")
        
        # Open and convert image to RGB
        img = Image.open(img_path).convert("RGB")
        
        # Apply standard transformations (resize, crop, normalize)
        input_tensor = preprocess(img).unsqueeze(0)  # Shape: (1, 3, 224, 224)
        
        # Forward pass through the network (no gradients needed for inference)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
        
        # Get Top-3 Predictions
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        
        print("  Top Predicted Labels:")
        for rank, (prob, class_idx) in enumerate(zip(top3_prob, top3_idx), 1):
            class_name = categories[class_idx]
            confidence = prob.item() * 100
            print(f"   {rank}. {class_name:<30} (Confidence: {confidence:.2f}%)")
        print()

    print("=" * 60)
    print(" Classification complete! ")
    print("=" * 60)

if __name__ == "__main__":
    main()
