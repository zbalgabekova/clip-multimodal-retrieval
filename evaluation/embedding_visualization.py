from pathlib import Path
import argparse

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TEST_CSV = ROOT / "data" / "processed" / "test.csv"

EMBEDDINGS_DIR = (
    ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "finetuned"
)

IMAGE_EMBEDDINGS_FILE = (
    EMBEDDINGS_DIR
    / "test_image_embeddings.pt"
)

TEXT_EMBEDDINGS_FILE = (
    EMBEDDINGS_DIR
    / "test_text_embeddings.pt"
)

OUTPUT_DIR = (
    ROOT
    / "evaluation"
    / "embedding_results"
)


# ============================================================
# Visualization settings
# ============================================================

DEFAULT_NUM_PAIRS = 500

RANDOM_SEED = 42

PERPLEXITY = 30

N_ITER = 1000


# ============================================================
# Utility
# ============================================================

def prepare_output_directory():

    """
    Create the output directory safely.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\nOutput directory:"
        f"\n{OUTPUT_DIR.resolve()}"
    )


# ============================================================
# Load dataset information
# ============================================================

def load_data():

    if not TEST_CSV.exists():

        raise FileNotFoundError(
            f"Test CSV not found:\n"
            f"{TEST_CSV}"
        )

    df = pd.read_csv(TEST_CSV)

    if "image" not in df.columns:

        raise ValueError(
            "test.csv does not contain "
            "an 'image' column."
        )

    if "caption" not in df.columns:

        raise ValueError(
            "test.csv does not contain "
            "a 'caption' column."
        )

    image_names = (
        df["image"]
        .drop_duplicates()
        .tolist()
    )

    captions = df["caption"].tolist()

    print(
        f"\nTest images: {len(image_names)}"
    )

    print(
        f"Test captions: {len(captions)}"
    )

    return (
        df,
        image_names,
        captions,
    )


# ============================================================
# Load embeddings
# ============================================================

def load_embeddings():

    if not IMAGE_EMBEDDINGS_FILE.exists():

        raise FileNotFoundError(
            f"\nImage embeddings not found:\n"
            f"{IMAGE_EMBEDDINGS_FILE}\n\n"
            "Run retrieval.py first."
        )

    if not TEXT_EMBEDDINGS_FILE.exists():

        raise FileNotFoundError(
            f"\nText embeddings not found:\n"
            f"{TEXT_EMBEDDINGS_FILE}\n\n"
            "Run retrieval.py first."
        )

    print(
        "\nLoading embeddings..."
    )

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

    if not isinstance(
        image_embeddings,
        torch.Tensor,
    ):

        raise TypeError(
            "Image embeddings are not "
            "a PyTorch tensor."
        )

    if not isinstance(
        text_embeddings,
        torch.Tensor,
    ):

        raise TypeError(
            "Text embeddings are not "
            "a PyTorch tensor."
        )

    print(
        f"Image embeddings shape: "
        f"{tuple(image_embeddings.shape)}"
    )

    print(
        f"Text embeddings shape: "
        f"{tuple(text_embeddings.shape)}"
    )

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    if image_embeddings.ndim != 2:

        raise ValueError(
            "Image embeddings must be 2D."
        )

    if text_embeddings.ndim != 2:

        raise ValueError(
            "Text embeddings must be 2D."
        )

    if image_embeddings.shape[1] != text_embeddings.shape[1]:

        raise ValueError(
            "Image and text embeddings "
            "have different dimensions."
        )

    return (
        image_embeddings,
        text_embeddings,
    )


# ============================================================
# Select paired samples
# ============================================================

def select_samples(
    image_embeddings,
    text_embeddings,
    num_pairs,
):

    num_images = len(
        image_embeddings
    )

    if num_pairs > num_images:

        print(
            f"\nRequested {num_pairs} pairs, "
            f"but only {num_images} images exist."
        )

        num_pairs = num_images

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    image_indices = rng.choice(
        num_images,
        size=num_pairs,
        replace=False,
    )

    image_indices = np.sort(
        image_indices
    )

    # --------------------------------------------------------
    # Flickr30K:
    #
    # image 0 -> captions 0-4
    # image 1 -> captions 5-9
    # image 2 -> captions 10-14
    #
    # We select the first caption for
    # every selected image.
    # --------------------------------------------------------

    text_indices = (
        image_indices * 5
    )

    selected_images = (
        image_embeddings[
            image_indices
        ]
    )

    selected_texts = (
        text_embeddings[
            text_indices
        ]
    )

    return (
        selected_images,
        selected_texts,
        image_indices,
        text_indices,
    )


# ============================================================
# Run t-SNE
# ============================================================

def run_tsne(
    image_embeddings,
    text_embeddings,
):

    print(
        "\nPreparing embeddings for t-SNE..."
    )

    # --------------------------------------------------------
    # Normalize embeddings.
    #
    # CLIP retrieval uses cosine similarity,
    # so we normalize before visualization.
    # --------------------------------------------------------

    image_embeddings = torch.nn.functional.normalize(
        image_embeddings,
        p=2,
        dim=1,
    )

    text_embeddings = torch.nn.functional.normalize(
        text_embeddings,
        p=2,
        dim=1,
    )

    # --------------------------------------------------------
    # Combine image + text embeddings
    # --------------------------------------------------------

    combined = torch.cat(
        [
            image_embeddings,
            text_embeddings,
        ],
        dim=0,
    )

    combined = combined.cpu().numpy()

    print(
        f"Total points: {len(combined)}"
    )

    print(
        f"Embedding dimension: "
        f"{combined.shape[1]}"
    )

    # --------------------------------------------------------
    # Make sure perplexity is valid
    # --------------------------------------------------------

    n_samples = len(combined)

    perplexity = min(
        PERPLEXITY,
        n_samples - 1,
    )

    if perplexity <= 1:

        raise ValueError(
            "Not enough samples for t-SNE."
        )

    print(
        f"\nRunning t-SNE..."
    )

    print(
        f"Perplexity: {perplexity}"
    )

    print(
        f"Iterations: {N_ITER}"
    )

    # --------------------------------------------------------
    # Cosine-distance t-SNE
    # --------------------------------------------------------

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=N_ITER,
        random_state=RANDOM_SEED,
        init="random",
        learning_rate="auto",
        metric="cosine",
    )

    embeddings_2d = tsne.fit_transform(
        combined
    )

    num_images = len(
        image_embeddings
    )

    image_2d = (
        embeddings_2d[:num_images]
    )

    text_2d = (
        embeddings_2d[num_images:]
    )

    print(
        "t-SNE complete."
    )

    return (
        image_2d,
        text_2d,
    )


# ============================================================
# Plot 1: Shared embedding space
# ============================================================

def plot_modality_space(
    image_2d,
    text_2d,
):

    print(
        "\nCreating shared embedding visualization..."
    )

    fig, ax = plt.subplots(
        figsize=(12, 9)
    )

    ax.scatter(
        image_2d[:, 0],
        image_2d[:, 1],
        s=30,
        alpha=0.65,
        label="Images",
    )

    ax.scatter(
        text_2d[:, 0],
        text_2d[:, 1],
        s=30,
        alpha=0.65,
        marker="^",
        label="Captions",
    )

    ax.set_title(
        "CLIP Shared Embedding Space",
        fontsize=18,
    )

    ax.set_xlabel(
        "t-SNE dimension 1",
        fontsize=12,
    )

    ax.set_ylabel(
        "t-SNE dimension 2",
        fontsize=12,
    )

    ax.legend(
        fontsize=11
    )

    ax.grid(
        alpha=0.15
    )

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR
        / "clip_embedding_space.png"
    )

    print(
        f"Saving figure to:\n"
        f"{output_file.resolve()}"
    )

    fig.savefig(
        str(output_file),
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved successfully."
    )


# ============================================================
# Plot 2: Image-caption alignment
# ============================================================

def plot_paired_embeddings(
    image_2d,
    text_2d,
):

    print(
        "\nCreating image-caption alignment visualization..."
    )

    fig, ax = plt.subplots(
        figsize=(12, 9)
    )

    num_pairs = len(
        image_2d
    )

    # --------------------------------------------------------
    # Draw connections
    # --------------------------------------------------------

    for i in range(num_pairs):

        ax.plot(
            [
                image_2d[i, 0],
                text_2d[i, 0],
            ],
            [
                image_2d[i, 1],
                text_2d[i, 1],
            ],
            linewidth=0.35,
            alpha=0.12,
        )

    # --------------------------------------------------------
    # Image points
    # --------------------------------------------------------

    ax.scatter(
        image_2d[:, 0],
        image_2d[:, 1],
        s=30,
        alpha=0.70,
        label="Image",
    )

    # --------------------------------------------------------
    # Caption points
    # --------------------------------------------------------

    ax.scatter(
        text_2d[:, 0],
        text_2d[:, 1],
        s=30,
        alpha=0.70,
        marker="^",
        label="Caption",
    )

    ax.set_title(
        "Image–Caption Alignment in CLIP Embedding Space",
        fontsize=18,
    )

    ax.set_xlabel(
        "t-SNE dimension 1",
        fontsize=12,
    )

    ax.set_ylabel(
        "t-SNE dimension 2",
        fontsize=12,
    )

    ax.legend(
        fontsize=11
    )

    ax.grid(
        alpha=0.15
    )

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR
        / "image_caption_alignment.png"
    )

    print(
        f"Saving figure to:\n"
        f"{output_file.resolve()}"
    )

    fig.savefig(
        str(output_file),
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved successfully."
    )


# ============================================================
# Plot 3: Image-only embedding space
# ============================================================

def plot_image_space(
    image_2d,
):

    print(
        "\nCreating image-only visualization..."
    )

    fig, ax = plt.subplots(
        figsize=(12, 9)
    )

    ax.scatter(
        image_2d[:, 0],
        image_2d[:, 1],
        s=30,
        alpha=0.70,
    )

    ax.set_title(
        "CLIP Image Embedding Space",
        fontsize=18,
    )

    ax.set_xlabel(
        "t-SNE dimension 1",
        fontsize=12,
    )

    ax.set_ylabel(
        "t-SNE dimension 2",
        fontsize=12,
    )

    ax.grid(
        alpha=0.15
    )

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR
        / "image_embedding_space.png"
    )

    print(
        f"Saving figure to:\n"
        f"{output_file.resolve()}"
    )

    fig.savefig(
        str(output_file),
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved successfully."
    )


# ============================================================
# Plot 4: Text-only embedding space
# ============================================================

def plot_text_space(
    text_2d,
):

    print(
        "\nCreating text-only visualization..."
    )

    fig, ax = plt.subplots(
        figsize=(12, 9)
    )

    ax.scatter(
        text_2d[:, 0],
        text_2d[:, 1],
        s=30,
        alpha=0.70,
        marker="^",
    )

    ax.set_title(
        "CLIP Text Embedding Space",
        fontsize=18,
    )

    ax.set_xlabel(
        "t-SNE dimension 1",
        fontsize=12,
    )

    ax.set_ylabel(
        "t-SNE dimension 2",
        fontsize=12,
    )

    ax.grid(
        alpha=0.15
    )

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR
        / "text_embedding_space.png"
    )

    print(
        f"Saving figure to:\n"
        f"{output_file.resolve()}"
    )

    fig.savefig(
        str(output_file),
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved successfully."
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Visualize CLIP image and "
            "text embeddings using t-SNE."
        )
    )

    parser.add_argument(
        "--num-pairs",
        type=int,
        default=DEFAULT_NUM_PAIRS,
        help=(
            "Number of image-caption pairs "
            "to visualize. Default: 500."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    if args.num_pairs < 2:

        raise ValueError(
            "--num-pairs must be at least 2."
        )

    print("=" * 70)
    print(
        "CLIP Embedding Visualization"
    )
    print("=" * 70)

    print(
        f"\nNumber of pairs: "
        f"{args.num_pairs}"
    )

    print(
        f"Random seed: "
        f"{RANDOM_SEED}"
    )

    print(
        f"Distance metric: "
        f"cosine"
    )

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

    prepare_output_directory()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    (
        df,
        image_names,
        captions,
    ) = load_data()

    # --------------------------------------------------------
    # Load embeddings
    # --------------------------------------------------------

    (
        image_embeddings,
        text_embeddings,
    ) = load_embeddings()

    # --------------------------------------------------------
    # Verify number of images
    # --------------------------------------------------------

    if len(image_embeddings) != len(image_names):

        raise ValueError(
            "\nMismatch between image embeddings "
            "and test images.\n\n"
            f"Image embeddings: "
            f"{len(image_embeddings)}\n"
            f"Test images: "
            f"{len(image_names)}\n\n"
            "Run retrieval.py again."
        )

    # --------------------------------------------------------
    # Verify number of captions
    # --------------------------------------------------------

    if len(text_embeddings) != len(captions):

        raise ValueError(
            "\nMismatch between text embeddings "
            "and test captions.\n\n"
            f"Text embeddings: "
            f"{len(text_embeddings)}\n"
            f"Test captions: "
            f"{len(captions)}\n\n"
            "Run retrieval.py again."
        )

    # --------------------------------------------------------
    # Select samples
    # --------------------------------------------------------

    (
        selected_images,
        selected_texts,
        image_indices,
        text_indices,
    ) = select_samples(
        image_embeddings,
        text_embeddings,
        args.num_pairs,
    )

    print(
        f"\nSelected pairs: "
        f"{len(selected_images)}"
    )

    # --------------------------------------------------------
    # t-SNE
    # --------------------------------------------------------

    (
        image_2d,
        text_2d,
    ) = run_tsne(
        selected_images,
        selected_texts,
    )

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------

    plot_modality_space(
        image_2d,
        text_2d,
    )

    plot_paired_embeddings(
        image_2d,
        text_2d,
    )

    plot_image_space(
        image_2d,
    )

    plot_text_space(
        text_2d,
    )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "Embedding visualization complete!"
    )

    print(
        "=" * 70
    )

    print(
        "\nGenerated files:"
    )

    print(
        OUTPUT_DIR
        / "clip_embedding_space.png"
    )

    print(
        OUTPUT_DIR
        / "image_caption_alignment.png"
    )

    print(
        OUTPUT_DIR
        / "image_embedding_space.png"
    )

    print(
        OUTPUT_DIR
        / "text_embedding_space.png"
    )


if __name__ == "__main__":
    main()