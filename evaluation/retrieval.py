from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import CLIPModel, AutoProcessor
from tqdm import tqdm

from evaluation.metrics import retrieval_metrics


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TEST_CSV = ROOT / "data" / "processed" / "test.csv"
IMAGE_DIR = ROOT / "data" / "raw" / "Images"

# Fine-tuned model
MODEL_PATH = ROOT / "checkpoints" / "best"

# Save fine-tuned embeddings separately
EMBEDDINGS_DIR = (
    ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "finetuned"
)

EMBEDDINGS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_EMBEDDINGS_FILE = (
    EMBEDDINGS_DIR / "test_image_embeddings.pt"
)

TEXT_EMBEDDINGS_FILE = (
    EMBEDDINGS_DIR / "test_text_embeddings.pt"
)

IMAGE_BATCH_SIZE = 32
TEXT_BATCH_SIZE = 32
SIMILARITY_CHUNK_SIZE = 128


# ============================================================
# Device
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# CLIP feature extraction
# ============================================================

def get_image_features(
    model,
    pixel_values,
):
    """
    Extract projected and normalized image embeddings.
    """

    vision_outputs = model.vision_model(
        pixel_values=pixel_values
    )

    image_features = vision_outputs.pooler_output

    image_features = model.visual_projection(
        image_features
    )

    image_features = F.normalize(
        image_features,
        p=2,
        dim=-1,
    )

    return image_features


def get_text_features(
    model,
    input_ids,
    attention_mask,
):
    """
    Extract projected and normalized text embeddings.
    """

    text_outputs = model.text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    text_features = text_outputs.pooler_output

    text_features = model.text_projection(
        text_features
    )

    text_features = F.normalize(
        text_features,
        p=2,
        dim=-1,
    )

    return text_features


# ============================================================
# Load test data
# ============================================================

def load_test_data():
    import pandas as pd

    df = pd.read_csv(TEST_CSV)

    # Keep image order consistent
    image_names = df["image"].drop_duplicates().tolist()

    # All captions in their original order
    captions = df["caption"].tolist()

    # Verify 5 captions per image
    captions_per_image = (
        df.groupby("image")
        .size()
    )

    if not (captions_per_image == 5).all():
        raise ValueError(
            "Not every test image has exactly 5 captions."
        )

    print(
        f"Test images: {len(image_names)}"
    )

    print(
        f"Test captions: {len(captions)}"
    )

    return df, image_names, captions


# ============================================================
# Encode images
# ============================================================

@torch.no_grad()
def encode_images(
    model,
    processor,
    image_names,
    device,
):
    from PIL import Image

    all_embeddings = []

    print("\nEncoding test images...")

    for start in tqdm(
        range(
            0,
            len(image_names),
            IMAGE_BATCH_SIZE,
        )
    ):
        batch_names = image_names[
            start:start + IMAGE_BATCH_SIZE
        ]

        images = []

        for image_name in batch_names:
            image_path = IMAGE_DIR / image_name

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            image = Image.open(
                image_path
            ).convert("RGB")

            images.append(image)

        inputs = processor(
            images=images,
            return_tensors="pt",
        )

        pixel_values = inputs[
            "pixel_values"
        ].to(
            device,
            non_blocking=True,
        )

        features = get_image_features(
            model,
            pixel_values,
        )

        all_embeddings.append(
            features.cpu()
        )

    embeddings = torch.cat(
        all_embeddings,
        dim=0,
    )

    return embeddings


# ============================================================
# Encode captions
# ============================================================

@torch.no_grad()
def encode_text(
    model,
    processor,
    captions,
    device,
):
    all_embeddings = []

    print("\nEncoding test captions...")

    for start in tqdm(
        range(
            0,
            len(captions),
            TEXT_BATCH_SIZE,
        )
    ):
        batch_captions = captions[
            start:start + TEXT_BATCH_SIZE
        ]

        inputs = processor(
            text=batch_captions,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs[
            "input_ids"
        ].to(device)

        attention_mask = inputs[
            "attention_mask"
        ].to(device)

        features = get_text_features(
            model,
            input_ids,
            attention_mask,
        )

        all_embeddings.append(
            features.cpu()
        )

    embeddings = torch.cat(
        all_embeddings,
        dim=0,
    )

    return embeddings


# ============================================================
# Text → Image retrieval
# ============================================================

def evaluate_text_to_image(
    text_embeddings,
    image_embeddings,
):
    """
    Evaluate caption → image retrieval.

    Each caption belongs to one image.
    Since every image has 5 captions:

        caption_index // 5 = image_index
    """

    num_texts = text_embeddings.shape[0]

    ground_truth = (
        torch.arange(num_texts)
        // 5
    )

    print("\n" + "=" * 60)
    print("Text → Image Retrieval")
    print("=" * 60)

    all_results = []

    for start in range(
        0,
        num_texts,
        SIMILARITY_CHUNK_SIZE,
    ):
        end = min(
            start + SIMILARITY_CHUNK_SIZE,
            num_texts,
        )

        text_chunk = text_embeddings[
            start:end
        ]

        similarity = (
            text_chunk
            @ image_embeddings.T
        )

        chunk_ground_truth = ground_truth[
            start:end
        ]

        metrics = retrieval_metrics(
            similarity,
            chunk_ground_truth,
            ks=(1, 5, 10),
        )

        all_results.append(
            (
                start,
                end,
                metrics,
            )
        )

    # Recalculate globally to avoid averaging
    # chunk percentages incorrectly.
    correct_at_1 = 0
    correct_at_5 = 0
    correct_at_10 = 0

    for start, end, _ in all_results:

        text_chunk = text_embeddings[
            start:end
        ]

        similarity = (
            text_chunk
            @ image_embeddings.T
        )

        gt = ground_truth[
            start:end
        ]

        rankings = torch.argsort(
            similarity,
            dim=1,
            descending=True,
        )

        correct_at_1 += (
            rankings[:, :1]
            == gt.unsqueeze(1)
        ).any(dim=1).sum().item()

        correct_at_5 += (
            rankings[:, :5]
            == gt.unsqueeze(1)
        ).any(dim=1).sum().item()

        correct_at_10 += (
            rankings[:, :10]
            == gt.unsqueeze(1)
        ).any(dim=1).sum().item()

    r1 = (
        correct_at_1
        / num_texts
        * 100
    )

    r5 = (
        correct_at_5
        / num_texts
        * 100
    )

    r10 = (
        correct_at_10
        / num_texts
        * 100
    )

    print(
        f"R@1:  {r1:.2f}%"
    )

    print(
        f"R@5:  {r5:.2f}%"
    )

    print(
        f"R@10: {r10:.2f}%"
    )

    return {
        "R@1": r1,
        "R@5": r5,
        "R@10": r10,
    }


# ============================================================
# Image → Text retrieval
# ============================================================

def evaluate_image_to_text(
    image_embeddings,
    text_embeddings,
):
    """
    Evaluate image → caption retrieval.

    Each image has 5 correct captions.

        image_index * 5
        through
        image_index * 5 + 4
    """

    num_images = image_embeddings.shape[0]

    print("\n" + "=" * 60)
    print("Image → Text Retrieval")
    print("=" * 60)

    correct_at_1 = 0
    correct_at_5 = 0
    correct_at_10 = 0

    for start in range(
        0,
        num_images,
        SIMILARITY_CHUNK_SIZE,
    ):
        end = min(
            start + SIMILARITY_CHUNK_SIZE,
            num_images,
        )

        image_chunk = image_embeddings[
            start:end
        ]

        similarity = (
            image_chunk
            @ text_embeddings.T
        )

        rankings = torch.argsort(
            similarity,
            dim=1,
            descending=True,
        )

        for local_index in range(
            end - start
        ):
            image_index = start + local_index

            correct_captions = torch.arange(
                image_index * 5,
                image_index * 5 + 5,
            )

            ranking = rankings[
                local_index
            ]

            correct_at_1 += int(
                torch.isin(
                    ranking[:1],
                    correct_captions,
                ).any()
            )

            correct_at_5 += int(
                torch.isin(
                    ranking[:5],
                    correct_captions,
                ).any()
            )

            correct_at_10 += int(
                torch.isin(
                    ranking[:10],
                    correct_captions,
                ).any()
            )

    r1 = (
        correct_at_1
        / num_images
        * 100
    )

    r5 = (
        correct_at_5
        / num_images
        * 100
    )

    r10 = (
        correct_at_10
        / num_images
        * 100
    )

    print(
        f"R@1:  {r1:.2f}%"
    )

    print(
        f"R@5:  {r5:.2f}%"
    )

    print(
        f"R@10: {r10:.2f}%"
    )

    return {
        "R@1": r1,
        "R@5": r5,
        "R@10": r10,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Flickr30K CLIP Retrieval Evaluation")
    print("=" * 60)

    print(
        f"\nModel: {MODEL_PATH}"
    )

    device = get_device()

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading fine-tuned CLIP...")

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH
    )

    model = CLIPModel.from_pretrained(
        MODEL_PATH
    )

    model = model.to(device)
    model.eval()

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    print("\nLoading test data...")

    (
        df,
        image_names,
        captions,
    ) = load_test_data()

    # --------------------------------------------------------
    # Encode images
    # --------------------------------------------------------

    image_embeddings = encode_images(
        model=model,
        processor=processor,
        image_names=image_names,
        device=device,
    )

    # --------------------------------------------------------
    # Encode text
    # --------------------------------------------------------

    text_embeddings = encode_text(
        model=model,
        processor=processor,
        captions=captions,
        device=device,
    )

    print(
        f"\nImage embedding shape: "
        f"{image_embeddings.shape}"
    )

    print(
        f"Text embedding shape: "
        f"{text_embeddings.shape}"
    )

    # --------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------

    torch.save(
        image_embeddings,
        IMAGE_EMBEDDINGS_FILE,
    )

    torch.save(
        text_embeddings,
        TEXT_EMBEDDINGS_FILE,
    )

    print(
        f"\nSaved image embeddings to:"
        f"\n{IMAGE_EMBEDDINGS_FILE}"
    )

    print(
        f"\nSaved text embeddings to:"
        f"\n{TEXT_EMBEDDINGS_FILE}"
    )

    # --------------------------------------------------------
    # Retrieval evaluation
    # --------------------------------------------------------

    text_to_image_results = (
        evaluate_text_to_image(
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
        )
    )

    image_to_text_results = (
        evaluate_image_to_text(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
        )
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Fine-Tuned CLIP Results")
    print("=" * 60)

    print("\nText → Image:")

    print(
        f"  R@1:  "
        f"{text_to_image_results['R@1']:.2f}%"
    )

    print(
        f"  R@5:  "
        f"{text_to_image_results['R@5']:.2f}%"
    )

    print(
        f"  R@10: "
        f"{text_to_image_results['R@10']:.2f}%"
    )

    print("\nImage → Text:")

    print(
        f"  R@1:  "
        f"{image_to_text_results['R@1']:.2f}%"
    )

    print(
        f"  R@5:  "
        f"{image_to_text_results['R@5']:.2f}%"
    )

    print(
        f"  R@10: "
        f"{image_to_text_results['R@10']:.2f}%"
    )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()