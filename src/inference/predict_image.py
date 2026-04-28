"""Run single-image inference with the Colab-trained EfficientNet baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

DEFAULT_LABELS = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
HIGH_RISK_LABELS = {"BCC", "MEL", "SCC"}
IMAGE_SIZE = 224


def build_model(num_classes: int):
    from torch import nn
    from torchvision import models

    # The checkpoint contains all trained weights; avoid any network download at inference time.
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_checkpoint(checkpoint_path: Path, device):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError(
            f"{checkpoint_path} does not look like a notebook checkpoint: "
            "missing model_state_dict"
        )
    return checkpoint


def make_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def predict_image(checkpoint_path: Path, image_path: Path, top_k: int = 3) -> dict[str, object]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(checkpoint_path, device)
    labels = checkpoint.get("labels", DEFAULT_LABELS)

    model = build_model(num_classes=len(labels)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = Image.open(image_path).convert("RGB")
    tensor = make_transform()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0).cpu()

    k = min(top_k, len(labels))
    scores, indices = torch.topk(probabilities, k=k)
    predictions = [
        {"label": labels[index], "probability": float(score)}
        for score, index in zip(scores.tolist(), indices.tolist())
    ]
    predicted_label = predictions[0]["label"]

    return {
        "image_path": str(image_path),
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "predicted_label": predicted_label,
        "predicted_triage_priority": "high" if predicted_label in HIGH_RISK_LABELS else "not_high",
        "top_k": predictions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_image(args.checkpoint_path, args.image_path, args.top_k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
