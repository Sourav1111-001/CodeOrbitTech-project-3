"""
==============================================
  Real-Life Image Classifier (MobileNetV2)
==============================================
How the pretrained model works:
  MobileNetV2 was trained on the ImageNet dataset containing 1.2 million
  images across 1000 categories (like dogs, cars, chairs, bananas, etc.).
  During training, the model learned to detect edges, textures, and shapes.
  When we pass a new image, it uses those learned patterns to make a
  prediction — without any retraining needed on our side.
  Think of it like an experienced doctor reading an X-ray at a glance.
"""

import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import requests
import io
import os
import cv2
from datetime import datetime

HEADERS      = {"User-Agent": "Mozilla/5.0"}
SAVE_DIR     = "saved_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Load pretrained model ---
print("\n[*] Loading pretrained model...")
weights    = MobileNet_V2_Weights.DEFAULT
model      = mobilenet_v2(weights=weights)
model.eval()
preprocess = weights.transforms()
categories = weights.meta["categories"]
print("[OK] Model ready!\n")  # already English

# ── Helpers ────────────────────────────────────────────────────────────────

def save_image(img: Image.Image, source: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = os.path.join(SAVE_DIR, f"{source}_{timestamp}.jpg")
    img.save(filename)
    return filename

def classify_image(img: Image.Image) -> list[tuple[str, float]]:
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    top3 = torch.topk(probs, 3)
    return [(categories[idx], round(prob.item() * 100, 2))
            for prob, idx in zip(top3.values, top3.indices)]

def show_results(results: list[tuple[str, float]], saved_path: str = None):
    print("\n  +------+-------------------------------------+------------+")
    print("  | Rank | Predicted Label                     | Confidence |")
    print("  +------+-------------------------------------+------------+")
    ranks = [" #1 ", " #2 ", " #3 "]
    for i, (label, conf) in enumerate(results):
        print(f"  | {ranks[i]} | {label:<35} | {conf:>8}%  |")
    print("  +------+-------------------------------------+------------+")
    if saved_path:
        print(f"  [Saved] Image saved to: {saved_path}\n")

# ── Option handlers ────────────────────────────────────────────────────────

def from_local_file():
    path = input("  Enter full image path: ").strip().strip('"')
    if not os.path.exists(path):
        print("  [!] File not found.")
        return
    img  = Image.open(path).convert("RGB")
    saved = save_image(img, "local")
    show_results(classify_image(img), saved)

def from_url():
    url = input("  Paste image URL: ").strip()
    try:
        r   = requests.get(url, timeout=10, headers=HEADERS)
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        saved = save_image(img, "url")
        show_results(classify_image(img), saved)
    except Exception as e:
        print(f"  [!] Error: {e}")

def from_samples():
    samples = [
        ("Dog_Golden",  "https://images.dog.ceo/breeds/retriever-golden/n02099601_3004.jpg"),
        ("Dog_Hound",   "https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg"),
        ("Dog_Labrador", "https://images.dog.ceo/breeds/labrador/n02099712_4323.jpg"),
    ]
    for name, url in samples:
        print(f"\n  [*] Classifying: {name}")
        try:
            img   = Image.open(io.BytesIO(requests.get(url, timeout=10, headers=HEADERS).content)).convert("RGB")
            saved = save_image(img, name)
            show_results(classify_image(img), saved)
        except Exception as e:
            print(f"  [!] Error: {e}")

def from_webcam():
    print("\n  [*] Opening webcam...")
    print("  >> Press SPACE to capture  |  Press ESC to go back\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [!] Webcam not found. Please check if it is connected.")
        return

    captured_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("  [!] Could not read frame.")
            break

        cv2.putText(frame, "SPACE = Capture  |  ESC = Back",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Webcam — Image Classifier", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:       # ESC
            break
        elif key == 32:     # SPACE
            captured_frame = frame.copy()
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_frame is not None:
        # OpenCV BGR -> PIL RGB
        img   = Image.fromarray(cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB))
        saved = save_image(img, "webcam")
        print("  [*] Photo captured! Classifying...")
        show_results(classify_image(img), saved)

# ── Main menu ──────────────────────────────────────────────────────────────

def run():
    while True:
        print("=" * 52)
        print("   Real-Life Image Classifier  --  MobileNetV2")
        print("=" * 52)
        print("  [1] Classify from a local image file")
        print("  [2] Classify from an internet URL")
        print("  [3] Run on sample images")
        print("  [4] Capture from webcam and classify")
        print("  [5] Exit")
        print(f"\n  (Saved images folder: {os.path.abspath(SAVE_DIR)})")

        choice = input("\n  Choose an option (1-5): ").strip()

        if   choice == "1": from_local_file()
        elif choice == "2": from_url()
        elif choice == "3": from_samples()
        elif choice == "4": from_webcam()
        elif choice == "5":
            print("\n  Goodbye!\n")
            break
        else:
            print("  [!] Invalid option. Please choose between 1 and 5.")

if __name__ == "__main__":
    run()
