from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from models.clip_model import load_clip


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "data" / "raw" / "Images"
TEST_FILE = ROOT / "data" / "processed" / "test.csv"

OUTPUT_DIR = ROOT / "data" / "processed" / "embeddings"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

IMAGE_BATCH_SIZE = 32
TEXT_BATCH_SIZE = 32

# Number of queries processed at once during similarity search.
SIMILARITY_BATCH_SIZE = 128

TOP_K = (1, 5, 10)


# ============================================================
# Load test data
# ============================================================

def load_test_data():
    """
    Load Flickr30K test captions.

    Returns:
        dataframe:
            All image-caption pairs.

        image_names:
            Unique image names in the test set.

        captions:
            All test captions.
    """

    df = pd.read_csv(TEST_FILE)

    if "image" not in df.columns:
        raise ValueError("Missing 'image' column.")

    if "caption" not in df.columns:
        raise ValueError("Missing 'caption' column.")

    # Preserve the original order.
    image_names = df["image"].drop_duplicates().tolist()

    captions = df["caption"].tolist()

    # --------------------------------------------------------
    # Verify that every image has exactly five captions.
    # --------------------------------------------------------

    caption_counts = df.groupby("image").size()

    invalid = caption_counts[caption_counts != 5]

    if len(invalid) > 0:
        raise ValueError(
            f"{len(invalid)} images do not have exactly 5 captions."
        )

    print(f"Test images: {len(image_names)}")
    print(f"Test captions: {len(captions)}")

    return df, image_names, captions


# ============================================================
# Encode images
# ============================================================

def encode_images(model, processor, image_names, device):
    """
    Encode all test images with CLIP.

    Returns:
        Tensor of shape:
            [num_images, embedding_dim]
    """

    embeddings = []

    print("\nEncoding images...")

    for start in tqdm(
        range(0, len(image_names), IMAGE_BATCH_SIZE)
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

            image = Image.open(image_path).convert("RGB")
            images.append(image)

        inputs = processor(
            images=images,
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values"].to(device)

        with torch.inference_mode():

            # Get the pooled visual representation.
            vision_outputs = model.vision_model(
                pixel_values=pixel_values
            )

            image_features = vision_outputs.pooler_output

            # Project to CLIP's shared embedding space.
            image_features = model.visual_projection(
                image_features
            )

        # Normalize embeddings.
        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True
        )

        embeddings.append(
            image_features.cpu()
        )

    embeddings = torch.cat(
        embeddings,
        dim=0
    )

    return embeddings


# ============================================================
# Encode text
# ============================================================

def encode_texts(model, processor, captions, device):
    """
    Encode all test captions with CLIP.

    Returns:
        Tensor of shape:
            [num_captions, embedding_dim]
    """

    embeddings = []

    print("\nEncoding captions...")

    for start in tqdm(
        range(0, len(captions), TEXT_BATCH_SIZE)
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

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.inference_mode():

            # Get the pooled text representation.
            text_outputs = model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            text_features = text_outputs.pooler_output

            # Project to CLIP's shared embedding space.
            text_features = model.text_projection(
                text_features
            )

        # Normalize embeddings.
        text_features = text_features / text_features.norm(
            dim=-1,
            keepdim=True
        )

        embeddings.append(
            text_features.cpu()
        )

    embeddings = torch.cat(
        embeddings,
        dim=0
    )

    return embeddings


# ============================================================
# Text → Image Retrieval
# ============================================================

def evaluate_text_to_image(
    text_embeddings,
    image_embeddings,
    num_images,
):
    """
    Evaluate Text → Image retrieval.

    Each caption has exactly one correct image.

    Caption ordering is:

        image 0 captions → indices 0-4
        image 1 captions → indices 5-9
        ...

    Returns:
        Dictionary containing Recall@1, Recall@5, Recall@10.
    """

    print("\n" + "=" * 60)
    print("Text → Image Retrieval")
    print("=" * 60)

    num_texts = text_embeddings.shape[0]

    correct = {
        k: 0
        for k in TOP_K
    }

    for start in tqdm(
        range(0, num_texts, SIMILARITY_BATCH_SIZE)
    ):

        text_batch = text_embeddings[
            start:start + SIMILARITY_BATCH_SIZE
        ]

        # [batch_texts, embedding_dim]
        #             ×
        # [embedding_dim, num_images]
        #
        # = [batch_texts, num_images]
        similarity = text_batch @ image_embeddings.T

        max_k = min(
            max(TOP_K),
            num_images
        )

        top_indices = torch.topk(
            similarity,
            k=max_k,
            dim=1,
        ).indices

        for local_idx in range(
            top_indices.shape[0]
        ):

            global_idx = start + local_idx

            # Five captions per image.
            ground_truth_image = global_idx // 5

            retrieved = top_indices[local_idx]

            for k in TOP_K:

                if ground_truth_image in retrieved[:k]:
                    correct[k] += 1

    results = {}

    for k in TOP_K:

        recall = (
            correct[k]
            / num_texts
            * 100.0
        )

        results[f"R@{k}"] = recall

    return results


# ============================================================
# Image → Text Retrieval
# ============================================================

def evaluate_image_to_text(
    image_embeddings,
    text_embeddings,
    num_images,
):
    """
    Evaluate Image → Text retrieval.

    Each image has five correct captions.

    Caption ordering is:

        image 0 → captions 0-4
        image 1 → captions 5-9
        ...

    Returns:
        Dictionary containing Recall@1, Recall@5, Recall@10.
    """

    print("\n" + "=" * 60)
    print("Image → Text Retrieval")
    print("=" * 60)

    num_texts = text_embeddings.shape[0]

    correct = {
        k: 0
        for k in TOP_K
    }

    for start in tqdm(
        range(0, num_images, SIMILARITY_BATCH_SIZE)
    ):

        image_batch = image_embeddings[
            start:start + SIMILARITY_BATCH_SIZE
        ]

        # [batch_images, embedding_dim]
        #             ×
        # [embedding_dim, num_texts]
        #
        # = [batch_images, num_texts]
        similarity = image_batch @ text_embeddings.T

        max_k = min(
            max(TOP_K),
            num_texts
        )

        top_indices = torch.topk(
            similarity,
            k=max_k,
            dim=1,
        ).indices

        for local_idx in range(
            top_indices.shape[0]
        ):

            global_image_idx = start + local_idx

            # Five correct captions.
            first_caption = global_image_idx * 5

            ground_truth = set(
                range(
                    first_caption,
                    first_caption + 5
                )
            )

            retrieved = top_indices[local_idx]

            for k in TOP_K:

                retrieved_k = set(
                    retrieved[:k].tolist()
                )

                if ground_truth.intersection(
                    retrieved_k
                ):
                    correct[k] += 1

    results = {}

    for k in TOP_K:

        recall = (
            correct[k]
            / num_images
            * 100.0
        )

        results[f"R@{k}"] = recall

    return results


# ============================================================
# Print results
# ============================================================

def print_results(
    text_to_image,
    image_to_text,
):
    print("\n")
    print("=" * 60)
    print("Flickr30K CLIP Retrieval Results")
    print("=" * 60)

    print("\nText → Image")

    for metric, value in text_to_image.items():

        print(
            f"  {metric:<5}: "
            f"{value:.2f}%"
        )

    print("\nImage → Text")

    for metric, value in image_to_text.items():

        print(
            f"  {metric:<5}: "
            f"{value:.2f}%"
        )

    print("\n" + "=" * 60)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Flickr30K CLIP Retrieval Evaluation")
    print("=" * 60)

    # --------------------------------------------------------
    # Load CLIP
    # --------------------------------------------------------

    model, processor, device = load_clip()

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    _, image_names, captions = load_test_data()

    num_images = len(image_names)

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
    # Encode captions
    # --------------------------------------------------------

    text_embeddings = encode_texts(
        model=model,
        processor=processor,
        captions=captions,
        device=device,
    )

    print("\nEmbedding shapes:")

    print(
        f"  Images:   {image_embeddings.shape}"
    )

    print(
        f"  Captions: {text_embeddings.shape}"
    )

    # --------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------

    image_embedding_file = (
        OUTPUT_DIR / "test_image_embeddings.pt"
    )

    text_embedding_file = (
        OUTPUT_DIR / "test_text_embeddings.pt"
    )

    torch.save(
        image_embeddings,
        image_embedding_file
    )

    torch.save(
        text_embeddings,
        text_embedding_file
    )

    print("\nEmbeddings saved:")
    print(f"  {image_embedding_file}")
    print(f"  {text_embedding_file}")

    # --------------------------------------------------------
    # Evaluate Text → Image
    # --------------------------------------------------------

    text_to_image = evaluate_text_to_image(
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
        num_images=num_images,
    )

    # --------------------------------------------------------
    # Evaluate Image → Text
    # --------------------------------------------------------

    image_to_text = evaluate_image_to_text(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        num_images=num_images,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_results(
        text_to_image=text_to_image,
        image_to_text=image_to_text,
    )


if __name__ == "__main__":
    main()