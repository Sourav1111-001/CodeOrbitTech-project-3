# ============================================================
#   Image Classification Using a Pretrained Model
#   Library : PyTorch + TorchVision
#   Model   : MobileNetV2 (pretrained on ImageNet)
# ============================================================
#
#   HOW THE PRETRAINED MODEL WORKS (Simple Explanation):
#   -----------------------------------------------------
#   MobileNetV2 is a neural network that was already trained
#   on 1.2 million images from 1000 different categories
#   (like dogs, cats, cars, chairs, etc.).
#
#   Think of it like a student who has already studied for
#   years. When you show it a new image, it uses everything
#   it already learned to make a prediction — no extra
#   training needed from our side!
#
#   Internally it works in 3 steps:
#     1. It scans the image for edges and colors (early layers)
#     2. It combines those into shapes and textures (mid layers)
#     3. It matches those shapes to a known category (final layer)
#
# ============================================================

import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import requests
from io import BytesIO

# ── Step 1: Load the pretrained model ───────────────────────
print("Loading pretrained MobileNetV2 model...")
weights    = MobileNet_V2_Weights.DEFAULT
model      = mobilenet_v2(weights=weights)
model.eval()                          # set to inference mode
preprocess = weights.transforms()     # official preprocessing
categories = weights.meta["categories"]  # 1000 ImageNet labels
print("Model ready!\n")

# ── Step 2: Define 3 sample images ──────────────────────────
SAMPLES = [
    {
        "name"  : "Golden Retriever (Dog)",
        "url"   : "https://images.dog.ceo/breeds/retriever-golden/n02099601_3004.jpg"
    },
    {
        "name"  : "Afghan Hound (Dog)",
        "url"   : "https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg"
    },
    {
        "name"  : "Cat",
        "url"   : None   # fetched live from Cat API
    },
]

# ── Step 3: Helper — download image from URL ────────────────
def download_image(url: str) -> Image.Image:
    response = requests.get(url, timeout=10,
                            headers={"User-Agent": "Mozilla/5.0"})
    return Image.open(BytesIO(response.content)).convert("RGB")

# ── Step 4: Helper — classify one image, return top 3 ───────
def classify(img: Image.Image) -> list:
    tensor = preprocess(img).unsqueeze(0)   # add batch dimension
    with torch.no_grad():                   # no gradient needed
        probs = torch.softmax(model(tensor), dim=1)[0]
    top3 = torch.topk(probs, 3)
    return [
        (categories[idx], round(prob.item() * 100, 2))
        for prob, idx in zip(top3.values, top3.indices)
    ]

# ── Step 5: Run classification on all 3 samples ─────────────
print("=" * 55)
print("   PREDICTED LABELS FOR SAMPLE IMAGES")
print("=" * 55)

for sample in SAMPLES:
    print(f"\n  Image : {sample['name']}")
    print(f"  {'-' * 45}")

    # Get image URL (Cat API gives a random URL)
    url = sample["url"]
    if url is None:
        resp = requests.get("https://api.thecatapi.com/v1/images/search",
                            timeout=10)
        url  = resp.json()[0]["url"]

    img     = download_image(url)
    results = classify(img)

    for rank, (label, confidence) in enumerate(results, start=1):
        bar = "#" * int(confidence / 5)   # simple text bar
        print(f"  #{rank}  {label:<30}  {confidence:>6}%  {bar}")

print("\n" + "=" * 55)
print("  Done! All 3 images classified successfully.")
print("=" * 55)
