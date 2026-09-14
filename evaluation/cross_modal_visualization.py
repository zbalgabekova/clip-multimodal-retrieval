"""
Cross-modal CLIP visualization.

Creates a heatmap of cosine similarity between selected
Flickr30K test images and their five ground-truth captions.

Run:
    python evaluation/cross_modal_visualization.py

Optional:
    python evaluation/cross_modal_visualization.py --num-images 30
    python evaluation/cross_modal_visualization.py --start-index 100
    python evaluation/cross_modal_visualization.py --num-images 20 --no-values
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent

TEST_CSV = ROOT / "data" / "processed" / "test.csv"

EMBEDDINGS_DIR = ROOT / "data" / "processed" / "embeddings" / "finetuned"

IMAGE_EMBEDDINGS_FILE = (
    EMBEDDINGS_DIR / "test_image_embeddings.pt"
)

TEXT_EMBEDDINGS_FILE = (
    EMBEDDINGS_DIR / "test_text_embeddings.pt"
)

OUTPUT_DIR = ROOT / "evaluation" / "embedding_results"

DEFAULT_NUM_IMAGES = 30
DEFAULT_START_INDEX = 0


def load_test_data():
    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Test CSV not found:\n{TEST_CSV}"
        )

    df = pd.read_csv(TEST_CSV)

    required = {"image", "caption"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"test.csv is missing columns: {missing}"
        )

    image_names = df["image"].drop_duplicates().tolist()
    captions = df["caption"].tolist()

    return image_names, captions


def load_embeddings(image_file, text_file):
    if not image_file.exists():
        raise FileNotFoundError(
            f"Image embeddings not found:\n{image_file}\n\n"
            "Run retrieval.py first."
        )

    if not text_file.exists():
        raise FileNotFoundError(
            f"Text embeddings not found:\n{text_file}\n\n"
            "Run retrieval.py first."
        )

    print("\nLoading embeddings...")

    image_embeddings = torch.load(
        image_file,
        map_location="cpu",
        weights_only=True,
    )

    text_embeddings = torch.load(
        text_file,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(image_embeddings, torch.Tensor):
        raise TypeError("Image embeddings are not a PyTorch tensor.")

    if not isinstance(text_embeddings, torch.Tensor):
        raise TypeError("Text embeddings are not a PyTorch tensor.")

    if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("Embeddings must have shape [N, D].")

    if image_embeddings.shape[1] != text_embeddings.shape[1]:
        raise ValueError(
            "Image and text embedding dimensions do not match."
        )

    print(f"Image embeddings: {tuple(image_embeddings.shape)}")
    print(f"Text embeddings: {tuple(text_embeddings.shape)}")

    return image_embeddings.float(), text_embeddings.float()


def select_samples(
    image_names,
    image_embeddings,
    text_embeddings,
    num_images,
    start_index,
):
    total_images = len(image_names)

    if start_index < 0 or start_index >= total_images:
        raise ValueError(
            f"Invalid --start-index {start_index}. "
            f"Valid range: 0-{total_images - 1}."
        )

    end_index = min(
        start_index + num_images,
        total_images,
    )

    image_indices = np.arange(
        start_index,
        end_index,
    )

    selected_names = [
        image_names[i]
        for i in image_indices
    ]

    # Flickr30K test.csv contains five consecutive captions
    # for every image.
    caption_indices = []

    for image_index in image_indices:
        caption_indices.extend(
            range(
                int(image_index) * 5,
                int(image_index) * 5 + 5,
            )
        )

    caption_indices = np.asarray(
        caption_indices,
        dtype=int,
    )

    selected_images = image_embeddings[image_indices]
    selected_texts = text_embeddings[caption_indices]

    return (
        selected_names,
        selected_images,
        selected_texts,
    )


def compute_similarity(image_embeddings, text_embeddings):
    image_embeddings = F.normalize(
        image_embeddings,
        p=2,
        dim=1,
    )

    text_embeddings = F.normalize(
        text_embeddings,
        p=2,
        dim=1,
    )

    return (
        image_embeddings @ text_embeddings.T
    ).cpu().numpy()


def print_summary(similarity, num_images):
    ground_truth = []

    for i in range(num_images):
        ground_truth.extend(
            similarity[i, i * 5:(i + 1) * 5]
        )

    ground_truth = np.asarray(ground_truth)

    non_ground_truth = []

    for i in range(num_images):
        gt_start = i * 5
        gt_end = gt_start + 5

        for j in range(similarity.shape[1]):
            if not (gt_start <= j < gt_end):
                non_ground_truth.append(
                    similarity[i, j]
                )

    non_ground_truth = np.asarray(non_ground_truth)

    print("\n" + "=" * 70)
    print("Cross-Modal Similarity Summary")
    print("=" * 70)

    print(
        f"Mean ground-truth similarity: "
        f"{ground_truth.mean():.4f}"
    )

    print(
        f"Mean non-ground-truth similarity: "
        f"{non_ground_truth.mean():.4f}"
    )

    print(
        f"Mean alignment gap: "
        f"{ground_truth.mean() - non_ground_truth.mean():.4f}"
    )

    print(
        f"Best ground-truth similarity: "
        f"{ground_truth.max():.4f}"
    )

    print(
        f"Worst ground-truth similarity: "
        f"{ground_truth.min():.4f}"
    )

    print("=" * 70)


def plot_heatmap(
    similarity,
    image_names,
    output_file,
    show_values=True,
):
    num_images = len(image_names)
    num_captions = similarity.shape[1]

    fig_width = max(
        14,
        min(24, num_captions * 0.18),
    )

    fig_height = max(
        9,
        min(18, num_images * 0.38),
    )

    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height)
    )

    im = ax.imshow(
        similarity,
        aspect="auto",
        interpolation="nearest",
    )

    # Each column is caption number 1-5 for an image.
    x_labels = []

    for image_number in range(num_images):
        for caption_number in range(5):
            x_labels.append(
                f"{image_number + 1}.{caption_number + 1}"
            )

    ax.set_xticks(
        np.arange(num_captions)
    )

    ax.set_xticklabels(
        x_labels,
        rotation=90,
        fontsize=7,
    )

    ax.set_xlabel(
        "Ground-truth caption (image.caption)"
    )

    y_labels = []

    for i, name in enumerate(image_names):
        stem = Path(name).stem

        if len(stem) > 18:
            stem = stem[:18] + "..."

        y_labels.append(
            f"{i + 1}: {stem}"
        )

    ax.set_yticks(
        np.arange(num_images)
    )

    ax.set_yticklabels(
        y_labels,
        fontsize=8,
    )

    ax.set_ylabel("Query image")

    # Separate the five captions belonging to each image.
    for x in range(4, num_captions, 5):
        ax.axvline(
            x + 0.5,
            linewidth=1.2,
        )

    # Highlight the correct five-caption block for each row.
    for y in range(num_images):
        rect = plt.Rectangle(
            (-0.5 + y * 5, y - 0.5),
            5,
            1,
            fill=False,
            linewidth=1.2,
        )

        # The x-position must correspond to this row's
        # five ground-truth captions.
        rect.set_x(y * 5 - 0.5)
        ax.add_patch(rect)

    if show_values and num_images <= 30:
        for i in range(num_images):
            for j in range(num_captions):
                ax.text(
                    j,
                    i,
                    f"{similarity[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                )

    ax.set_title(
        "Cross-Modal Image–Text Similarity",
        fontsize=18,
        pad=15,
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.025,
        pad=0.02,
    )

    cbar.set_label("Cosine similarity")

    fig.tight_layout()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\nSaving heatmap to:\n"
        f"{output_file.resolve()}"
    )

    fig.savefig(
        str(output_file),
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved successfully.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize cross-modal CLIP "
            "image-text similarity."
        )
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help="Number of images to visualize. Default: 30.",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=DEFAULT_START_INDEX,
        help="Starting test image index. Default: 0.",
    )

    parser.add_argument(
        "--no-values",
        action="store_true",
        help="Do not display numeric similarity values.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(
            OUTPUT_DIR
            / "cross_modal_similarity_heatmap.png"
        ),
        help="Output PNG path.",
    )

    args = parser.parse_args()

    if args.num_images < 2:
        raise ValueError(
            "--num-images must be at least 2."
        )

    print("=" * 70)
    print("CLIP Cross-Modal Visualization")
    print("=" * 70)

    print(
        f"\nImages to visualize: {args.num_images}"
    )

    print(
        f"Starting index: {args.start_index}"
    )

    image_names, captions = load_test_data()

    image_embeddings, text_embeddings = load_embeddings(
        IMAGE_EMBEDDINGS_FILE,
        TEXT_EMBEDDINGS_FILE,
    )

    if len(image_names) != len(image_embeddings):
        raise ValueError(
            "Number of image embeddings does not match "
            "number of test images."
        )

    if len(captions) != len(text_embeddings):
        raise ValueError(
            "Number of text embeddings does not match "
            "number of test captions."
        )

    (
        selected_names,
        selected_images,
        selected_texts,
    ) = select_samples(
        image_names,
        image_embeddings,
        text_embeddings,
        args.num_images,
        args.start_index,
    )

    print(
        f"\nSelected images: {len(selected_names)}"
    )

    print(
        f"Selected captions: {len(selected_names) * 5}"
    )

    print(
        "\nComputing image-text cosine similarity..."
    )

    similarity = compute_similarity(
        selected_images,
        selected_texts,
    )

    print(
        f"Similarity matrix: {similarity.shape}"
    )

    print_summary(
        similarity,
        len(selected_names),
    )

    output_file = Path(args.output)

    plot_heatmap(
        similarity=similarity,
        image_names=selected_names,
        output_file=output_file,
        show_values=not args.no_values,
    )

    print("\n" + "=" * 70)
    print("Cross-modal visualization complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
