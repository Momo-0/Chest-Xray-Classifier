import torch
from torchvision import transforms, datasets
from PIL import Image
import os
import sys


# ──────────────────────────────────────────────
#  Preprocessing
# ──────────────────────────────────────────────

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ──────────────────────────────────────────────
#  Image Loading
# ──────────────────────────────────────────────

def load_image(image_path: str) -> torch.Tensor:
    """
    Load an X-ray image from disk, convert to RGB if needed,
    apply preprocessing, and return a batched tensor.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path)

    # X-rays are often grayscale — convert to RGB for the model
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = preprocess(image)
    return image.unsqueeze(0)   # add batch dimension


# ──────────────────────────────────────────────
#  Model Loading
# ──────────────────────────────────────────────

def load_model(model_path: str, device: torch.device):
    """
    Load the full saved model (architecture + weights).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = torch.load(model_path, map_location=device)
    model.eval()
    return model.to(device)


# ──────────────────────────────────────────────
#  Class Names
# ──────────────────────────────────────────────

def get_class_names(data_dir: str) -> list:
    """
    Derive class names from the training folder structure.
    Expected: data_dir/train/<COVID | Normal | Pneumonia>
    """
    train_path = os.path.join(data_dir, 'train')
    dataset = datasets.ImageFolder(train_path)
    return dataset.classes


# ──────────────────────────────────────────────
#  Inference
# ──────────────────────────────────────────────

def predict(image_path: str, model_path: str, data_dir: str) -> dict:
    """
    Run inference on a single chest X-ray image.

    Returns a dict with:
        predicted_class  : str   — e.g. 'COVID', 'Normal', 'Pneumonia'
        confidence       : float — probability of the predicted class (0–1)
        all_probs        : dict  — {class_name: probability} for all classes
    """
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = get_class_names(data_dir)
    model       = load_model(model_path, device)
    image       = load_image(image_path).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs   = torch.softmax(outputs, dim=1).squeeze()
        pred_idx = torch.argmax(probs).item()

    predicted_class = class_names[pred_idx]
    confidence      = probs[pred_idx].item()
    all_probs       = {cls: round(probs[i].item(), 4)
                       for i, cls in enumerate(class_names)}

    return {
        'predicted_class': predicted_class,
        'confidence':      round(confidence, 4),
        'all_probs':       all_probs,
    }


# ──────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────

if __name__ == '__main__':
    # ── Configuration ──────────────────────────
    DATA_DIR    = '/home/ahmed/Downloads'                          # ← dataset root
    MODEL_PATH  = 'saved_models/best_model.pth'                   # ← trained model
    IMAGE_PATH  = (sys.argv[1] if len(sys.argv) > 1
                   else '/home/ahmed/Downloads/test/Normal/Normal-16.png')
    # ───────────────────────────────────────────

    result = predict(IMAGE_PATH, MODEL_PATH, DATA_DIR)

    print(f"\nImage      : {IMAGE_PATH}")
    print(f"Prediction : {result['predicted_class']}")
    print(f"Confidence : {result['confidence'] * 100:.2f}%")
    print("\nClass probabilities:")
    for cls, prob in result['all_probs'].items():
        bar = '█' * int(prob * 30)
        print(f"  {cls:<12} {prob * 100:5.2f}%  {bar}")
