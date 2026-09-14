from pathlib import Path

import torch
from PIL import Image

from models.clip_model import load_clip


ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "data" / "raw" / "Images"


def main():

    model, processor, device = load_clip()

    # --------------------------------------------------
    # Select an image
    # --------------------------------------------------

    image_path = IMAGE_DIR / "1000092795.jpg"

    image = Image.open(image_path).convert("RGB")

    # --------------------------------------------------
    # Candidate classes
    # --------------------------------------------------

    labels = [
        "a photo of a person",
        "a photo of a dog",
        "a photo of a cat",
        "a photo of a car",
        "a photo of a bicycle",
        "a photo of a horse",
    ]

    # --------------------------------------------------
    # CLIP preprocessing
    # --------------------------------------------------

    inputs = processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------

    with torch.inference_mode():

        outputs = model(**inputs)

        logits = outputs.logits_per_image

        probabilities = logits.softmax(dim=1)

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    probabilities = probabilities[0]

    ranked = torch.argsort(
        probabilities,
        descending=True
    )

    print("\nZero-shot classification:")
    print("-" * 50)

    for index in ranked:

        label = labels[index]
        probability = probabilities[index]

        print(
            f"{label:30s} "
            f"{probability.item() * 100:6.2f}%"
        )


if __name__ == "__main__":
    main()