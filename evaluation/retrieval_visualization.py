from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "data" / "raw" / "Images"
TEST_FILE = ROOT / "data" / "processed" / "test.csv"

EMBEDDING_DIR = (
    ROOT / "data" / "processed" / "embeddings"
)

IMAGE_EMBEDDINGS_FILE = (
    EMBEDDING_DIR / "test_image_embeddings.pt"
)

TEXT_EMBEDDINGS_FILE = (
    EMBEDDING_DIR / "test_text_embeddings.pt"
)


# ============================================================
# Configuration
# ============================================================

TOP_K = 5


# ============================================================
# Load data
# ============================================================

def load_data():
    """
    Load test captions and precomputed CLIP embeddings.
    """

    print("Loading test data...")

    df = pd.read_csv(TEST_FILE)

    image_embeddings = torch.load(
        IMAGE_EMBEDDINGS_FILE,
        map_location="cpu",
        weights_only=True,
    )

    text_embeddings = torch.load(
        TEXT_EMBEDDINGS_FILE,
        map_location="cpu",
        weights_only=True,
    )

    # Unique images in the same order used during retrieval.
    image_names = (
        df["image"]
        .drop_duplicates()
        .tolist()
    )

    captions = df["caption"].tolist()

    print(f"Images: {len(image_names)}")
    print(f"Captions: {len(captions)}")

    return (
        df,
        image_names,
        captions,
        image_embeddings,
        text_embeddings,
    )


# ============================================================
# Text → Image Retrieval
# ============================================================

def text_to_image(
    query_index,
    image_names,
    captions,
    image_embeddings,
    text_embeddings,
):
    """
    Retrieve images for a selected caption.

    Args:
        query_index:
            Index of the caption in the test captions.

    Returns:
        Top-k image indices and similarity scores.
    """

    if query_index < 0 or query_index >= len(captions):
        raise IndexError(
            f"query_index must be between "
            f"0 and {len(captions) - 1}"
        )

    query = text_embeddings[query_index]

    # Since embeddings are already normalized,
    # dot product = cosine similarity.
    similarity = query @ image_embeddings.T

    scores, indices = torch.topk(
        similarity,
        k=min(TOP_K, len(image_names)),
    )

    return (
        captions[query_index],
        indices.tolist(),
        scores.tolist(),
    )


# ============================================================
# Image → Text Retrieval
# ============================================================

def image_to_text(
    image_index,
    image_names,
    captions,
    image_embeddings,
    text_embeddings,
):
    """
    Retrieve captions for a selected image.

    Args:
        image_index:
            Index of the image in the test set.

    Returns:
        Top-k caption indices and similarity scores.
    """

    if image_index < 0 or image_index >= len(image_names):
        raise IndexError(
            f"image_index must be between "
            f"0 and {len(image_names) - 1}"
        )

    query = image_embeddings[image_index]

    similarity = query @ text_embeddings.T

    scores, indices = torch.topk(
        similarity,
        k=min(TOP_K, len(captions)),
    )

    return (
        indices.tolist(),
        scores.tolist(),
    )


# ============================================================
# Check Text → Image Ground Truth
# ============================================================

def is_correct_text_to_image(
    query_index,
    image_index,
):
    """
    Check whether a retrieved image is the
    correct image for a caption.

    Every five consecutive captions belong
    to the same image.
    """

    ground_truth_image = query_index // 5

    return image_index == ground_truth_image


# ============================================================
# Check Image → Text Ground Truth
# ============================================================

def is_correct_image_to_text(
    image_index,
    caption_index,
):
    """
    Check whether a retrieved caption belongs
    to the selected image.

    Each image has exactly five captions.
    """

    first_caption = image_index * 5

    last_caption = first_caption + 5

    return first_caption <= caption_index < last_caption


# ============================================================
# Plot Text → Image
# ============================================================

def plot_text_to_image(
    query,
    image_indices,
    scores,
    image_names,
    query_index,
):
    """
    Display the query caption and top-k retrieved images.
    """

    fig, axes = plt.subplots(
        1,
        len(image_indices),
        figsize=(18, 4),
    )

    if len(image_indices) == 1:
        axes = [axes]

    fig.suptitle(
        "Text → Image Retrieval",
        fontsize=16,
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.02,
        f'Query: "{query}"',
        ha="center",
        fontsize=11,
    )

    for rank, (
        ax,
        image_index,
        score,
    ) in enumerate(
        zip(
            axes,
            image_indices,
            scores,
        ),
        start=1,
    ):

        image_name = image_names[image_index]

        image_path = IMAGE_DIR / image_name

        image = Image.open(
            image_path
        ).convert("RGB")

        ax.imshow(image)
        ax.axis("off")

        correct = is_correct_text_to_image(
            query_index,
            image_index,
        )

        label = (
            f"Rank {rank}\n"
            f"Similarity: {score:.4f}"
        )

        if correct:
            label += "\n✓ Correct"

        ax.set_title(
            label,
            fontsize=10,
        )

    plt.tight_layout(
        rect=[0, 0.08, 1, 0.92]
    )

    output_file = (
        ROOT
        / "evaluation"
        / "text_to_image_result.png"
    )

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    print(
        f"\nVisualization saved to:\n"
        f"{output_file}"
    )

    plt.show()


# ============================================================
# Plot Image → Text
# ============================================================

def plot_image_to_text(
    image_index,
    caption_indices,
    scores,
    image_names,
    captions,
):
    """
    Display an image and its top-k retrieved captions.
    """

    image_name = image_names[image_index]

    image_path = IMAGE_DIR / image_name

    image = Image.open(
        image_path
    ).convert("RGB")

    fig = plt.figure(
        figsize=(12, 8)
    )

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    ax_image = fig.add_axes(
        [0.05, 0.35, 0.40, 0.55]
    )

    ax_image.imshow(image)
    ax_image.axis("off")

    ax_image.set_title(
        f"Image: {image_name}",
        fontsize=12,
        fontweight="bold",
    )

    # --------------------------------------------------------
    # Captions
    # --------------------------------------------------------

    ax_text = fig.add_axes(
        [0.50, 0.05, 0.47, 0.85]
    )

    ax_text.axis("off")

    lines = [
        "Top-5 Retrieved Captions",
        "",
    ]

    for rank, (
        caption_index,
        score,
    ) in enumerate(
        zip(
            caption_indices,
            scores,
        ),
        start=1,
    ):

        caption = captions[caption_index]

        correct = is_correct_image_to_text(
            image_index,
            caption_index,
        )

        marker = "✓" if correct else " "

        lines.append(
            f"{marker} Rank {rank} "
            f"(similarity: {score:.4f})"
        )

        lines.append(
            f'"{caption}"'
        )

        lines.append("")

    ax_text.text(
        0,
        1,
        "\n".join(lines),
        va="top",
        fontsize=10,
        wrap=True,
    )

    fig.suptitle(
        "Image → Text Retrieval",
        fontsize=16,
        fontweight="bold",
    )

    output_file = (
        ROOT
        / "evaluation"
        / "image_to_text_result.png"
    )

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    print(
        f"\nVisualization saved to:\n"
        f"{output_file}"
    )

    plt.show()


# ============================================================
# Print examples
# ============================================================

def print_text_to_image_results(
    query,
    image_indices,
    scores,
    image_names,
    query_index,
):
    """
    Print Text → Image results to terminal.
    """

    print("\n" + "=" * 70)
    print("TEXT → IMAGE RETRIEVAL")
    print("=" * 70)

    print(f'\nQuery:\n"{query}"\n')

    for rank, (
        image_index,
        score,
    ) in enumerate(
        zip(
            image_indices,
            scores,
        ),
        start=1,
    ):

        correct = is_correct_text_to_image(
            query_index,
            image_index,
        )

        marker = "✓" if correct else " "

        print(
            f"{marker} Rank {rank}: "
            f"{image_names[image_index]} "
            f"(similarity={score:.4f})"
        )

    print("=" * 70)


def print_image_to_text_results(
    image_index,
    caption_indices,
    scores,
    image_names,
    captions,
):
    """
    Print Image → Text results to terminal.
    """

    print("\n" + "=" * 70)
    print("IMAGE → TEXT RETRIEVAL")
    print("=" * 70)

    print(
        f"\nImage:\n"
        f"{image_names[image_index]}\n"
    )

    for rank, (
        caption_index,
        score,
    ) in enumerate(
        zip(
            caption_indices,
            scores,
        ),
        start=1,
    ):

        correct = is_correct_image_to_text(
            image_index,
            caption_index,
        )

        marker = "✓" if correct else " "

        print(
            f"{marker} Rank {rank}: "
            f"(similarity={score:.4f})"
        )

        print(
            f'    "{captions[caption_index]}"'
        )

    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Visualize CLIP Flickr30K "
            "retrieval results."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "text-to-image",
            "image-to-text",
        ],
        default="text-to-image",
        help="Retrieval direction.",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help=(
            "Caption index for text-to-image "
            "or image index for image-to-text."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        df,
        image_names,
        captions,
        image_embeddings,
        text_embeddings,
    ) = load_data()

    # --------------------------------------------------------
    # Text → Image
    # --------------------------------------------------------

    if args.mode == "text-to-image":

        (
            query,
            image_indices,
            scores,
        ) = text_to_image(
            query_index=args.index,
            image_names=image_names,
            captions=captions,
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
        )

        print_text_to_image_results(
            query=query,
            image_indices=image_indices,
            scores=scores,
            image_names=image_names,
            query_index=args.index,
        )

        plot_text_to_image(
            query=query,
            image_indices=image_indices,
            scores=scores,
            image_names=image_names,
            query_index=args.index,
        )

    # --------------------------------------------------------
    # Image → Text
    # --------------------------------------------------------

    elif args.mode == "image-to-text":

        (
            caption_indices,
            scores,
        ) = image_to_text(
            image_index=args.index,
            image_names=image_names,
            captions=captions,
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
        )

        print_image_to_text_results(
            image_index=args.index,
            caption_indices=caption_indices,
            scores=scores,
            image_names=image_names,
            captions=captions,
        )

        plot_image_to_text(
            image_index=args.index,
            caption_indices=caption_indices,
            scores=scores,
            image_names=image_names,
            captions=captions,
        )


if __name__ == "__main__":
    main()