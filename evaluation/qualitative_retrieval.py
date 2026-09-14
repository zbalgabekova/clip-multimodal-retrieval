from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from transformers import CLIPModel, AutoProcessor


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TEST_CSV = ROOT / "data" / "processed" / "test.csv"
IMAGE_DIR = ROOT / "data" / "raw" / "Images"

MODEL_PATH = ROOT / "checkpoints" / "best"

EMBEDDINGS_DIR = (
    ROOT
    / "data"
    / "processed"
    / "embeddings"
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
    / "qualitative_results"
)

TOP_K = 5

IMAGE_BATCH_SIZE = 32
TEXT_BATCH_SIZE = 32


# ============================================================
# Device
# ============================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# Load test data
# ============================================================

def load_test_data():

    df = pd.read_csv(TEST_CSV)

    image_names = (
        df["image"]
        .drop_duplicates()
        .tolist()
    )

    captions = df["caption"].tolist()

    # --------------------------------------------------------
    # Verify Flickr30K structure
    # --------------------------------------------------------

    counts = df.groupby("image").size()

    invalid = counts[counts != 5]

    if len(invalid) > 0:

        raise ValueError(
            "Some images do not have exactly "
            "5 captions."
        )

    return df, image_names, captions


# ============================================================
# Feature extraction
# ============================================================

@torch.no_grad()
def get_image_features(
    model,
    pixel_values,
):

    vision_outputs = model.vision_model(
        pixel_values=pixel_values
    )

    image_features = (
        vision_outputs.pooler_output
    )

    image_features = (
        model.visual_projection(
            image_features
        )
    )

    image_features = F.normalize(
        image_features,
        p=2,
        dim=-1,
    )

    return image_features


@torch.no_grad()
def get_text_features(
    model,
    input_ids,
    attention_mask,
):

    text_outputs = model.text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    text_features = (
        text_outputs.pooler_output
    )

    text_features = (
        model.text_projection(
            text_features
        )
    )

    text_features = F.normalize(
        text_features,
        p=2,
        dim=-1,
    )

    return text_features


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

    embeddings = []

    print("\nEncoding test images...")

    for start in range(
        0,
        len(image_names),
        IMAGE_BATCH_SIZE,
    ):

        batch_names = image_names[
            start:start + IMAGE_BATCH_SIZE
        ]

        images = []

        for image_name in batch_names:

            image_path = (
                IMAGE_DIR / image_name
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
        ].to(device)

        features = get_image_features(
            model,
            pixel_values,
        )

        embeddings.append(
            features.cpu()
        )

        print(
            f"\rImages: "
            f"{min(start + IMAGE_BATCH_SIZE, len(image_names))}"
            f"/{len(image_names)}",
            end="",
        )

    print()

    return torch.cat(
        embeddings,
        dim=0,
    )


# ============================================================
# Encode captions
# ============================================================

@torch.no_grad()
def encode_texts(
    model,
    processor,
    captions,
    device,
):

    embeddings = []

    print("\nEncoding test captions...")

    for start in range(
        0,
        len(captions),
        TEXT_BATCH_SIZE,
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

        embeddings.append(
            features.cpu()
        )

        print(
            f"\rCaptions: "
            f"{min(start + TEXT_BATCH_SIZE, len(captions))}"
            f"/{len(captions)}",
            end="",
        )

    print()

    return torch.cat(
        embeddings,
        dim=0,
    )


# ============================================================
# Load or create embeddings
# ============================================================

def get_embeddings(
    model,
    processor,
    image_names,
    captions,
    device,
    recompute=False,
):

    EMBEDDINGS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings_exist = (
        IMAGE_EMBEDDINGS_FILE.exists()
        and TEXT_EMBEDDINGS_FILE.exists()
    )

    if embeddings_exist and not recompute:

        print(
            "\nLoading cached embeddings..."
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

        print(
            "Cached embeddings loaded."
        )

        return (
            image_embeddings,
            text_embeddings,
        )

    # --------------------------------------------------------
    # Recompute
    # --------------------------------------------------------

    image_embeddings = encode_images(
        model=model,
        processor=processor,
        image_names=image_names,
        device=device,
    )

    text_embeddings = encode_texts(
        model=model,
        processor=processor,
        captions=captions,
        device=device,
    )

    torch.save(
        image_embeddings,
        IMAGE_EMBEDDINGS_FILE,
    )

    torch.save(
        text_embeddings,
        TEXT_EMBEDDINGS_FILE,
    )

    print(
        "\nEmbeddings saved."
    )

    return (
        image_embeddings,
        text_embeddings,
    )


# ============================================================
# Text → Image ranking
# ============================================================

def rank_text_to_image(
    caption_index,
    text_embeddings,
    image_embeddings,
):

    query_embedding = (
        text_embeddings[caption_index]
    )

    similarity = (
        query_embedding
        @ image_embeddings.T
    )

    scores, indices = torch.topk(
        similarity,
        k=TOP_K,
    )

    correct_image_index = (
        caption_index // 5
    )

    return (
        scores,
        indices,
        correct_image_index,
    )


# ============================================================
# Image → Text ranking
# ============================================================

def rank_image_to_text(
    image_index,
    image_embeddings,
    text_embeddings,
):

    query_embedding = (
        image_embeddings[image_index]
    )

    similarity = (
        query_embedding
        @ text_embeddings.T
    )

    scores, indices = torch.topk(
        similarity,
        k=TOP_K,
    )

    correct_start = (
        image_index * 5
    )

    correct_end = (
        correct_start + 5
    )

    correct_caption_indices = set(
        range(
            correct_start,
            correct_end,
        )
    )

    return (
        scores,
        indices,
        correct_caption_indices,
    )


# ============================================================
# Find qualitative examples
# ============================================================

def find_text_to_image_examples(
    text_embeddings,
    image_embeddings,
    num_examples=1,
):

    perfect = []
    hard = []
    failure = []

    for caption_index in range(
        len(text_embeddings)
    ):

        (
            scores,
            indices,
            correct_index,
        ) = rank_text_to_image(
            caption_index,
            text_embeddings,
            image_embeddings,
        )

        ranked = [
            index.item()
            for index in indices
        ]

        if ranked[0] == correct_index:

            perfect.append(
                caption_index
            )

        elif correct_index in ranked:

            hard.append(
                caption_index
            )

        else:

            failure.append(
                caption_index
            )

        if (
            len(perfect) >= num_examples
            and len(hard) >= num_examples
            and len(failure) >= num_examples
        ):
            break

    return {
        "perfect": perfect[:num_examples],
        "hard": hard[:num_examples],
        "failure": failure[:num_examples],
    }


def find_image_to_text_examples(
    image_embeddings,
    text_embeddings,
    num_examples=1,
):

    perfect = []
    partial = []
    failure = []

    for image_index in range(
        len(image_embeddings)
    ):

        (
            scores,
            indices,
            correct_indices,
        ) = rank_image_to_text(
            image_index,
            image_embeddings,
            text_embeddings,
        )

        retrieved = {
            index.item()
            for index in indices
        }

        num_correct = len(
            retrieved
            & correct_indices
        )

        if num_correct >= 4:

            perfect.append(
                image_index
            )

        elif num_correct > 0:

            partial.append(
                image_index
            )

        else:

            failure.append(
                image_index
            )

        if (
            len(perfect) >= num_examples
            and len(partial) >= num_examples
            and len(failure) >= num_examples
        ):
            break

    return {
        "perfect": perfect[:num_examples],
        "partial": partial[:num_examples],
        "failure": failure[:num_examples],
    }


# ============================================================
# Text → Image visualization
# ============================================================

def visualize_text_to_image(
    caption_index,
    captions,
    image_names,
    text_embeddings,
    image_embeddings,
    category,
):

    (
        scores,
        indices,
        correct_index,
    ) = rank_text_to_image(
        caption_index,
        text_embeddings,
        image_embeddings,
    )

    query = captions[
        caption_index
    ]

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        TOP_K,
        figsize=(18, 4.5),
    )

    if TOP_K == 1:
        axes = [axes]

    for rank, (
        ax,
        score,
        image_index,
    ) in enumerate(
        zip(
            axes,
            scores,
            indices,
        ),
        start=1,
    ):

        image_index = (
            image_index.item()
        )

        image_name = (
            image_names[image_index]
        )

        image_path = (
            IMAGE_DIR / image_name
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        ax.imshow(image)

        is_correct = (
            image_index
            == correct_index
        )

        if is_correct:

            label = "✓ Ground Truth"

        else:

            label = "✗ Not Ground Truth"

        ax.set_title(
            f"Rank #{rank}\n"
            f"{label}\n"
            f"Similarity: {score.item():.3f}",
            fontsize=10,
        )

        ax.axis("off")

    fig.suptitle(
        "Text → Image Retrieval",
        fontsize=17,
        y=1.04,
    )

    fig.text(
        0.5,
        0.97,
        f'Query: "{query}"',
        ha="center",
        va="top",
        fontsize=11,
    )

    plt.tight_layout()

    output_dir = (
        OUTPUT_DIR
        / "text_to_image"
        / category
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / f"caption_{caption_index}.png"
    )

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# Image → Text visualization
# ============================================================

def visualize_image_to_text(
    image_index,
    image_names,
    captions,
    image_embeddings,
    text_embeddings,
    category,
):

    (
        scores,
        indices,
        correct_indices,
    ) = rank_image_to_text(
        image_index,
        image_embeddings,
        text_embeddings,
    )

    image_name = (
        image_names[image_index]
    )

    image_path = (
        IMAGE_DIR / image_name
    )

    query_image = Image.open(
        image_path
    ).convert("RGB")

    # --------------------------------------------------------
    # Count correct results
    # --------------------------------------------------------

    retrieved_indices = [
        index.item()
        for index in indices
    ]

    num_correct = len(
        set(retrieved_indices)
        & correct_indices
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(14, 8)
    )

    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1, 1.6],
    )

    # --------------------------------------------------------
    # Query image
    # --------------------------------------------------------

    ax_image = fig.add_subplot(
        grid[0, 0]
    )

    ax_image.imshow(
        query_image
    )

    ax_image.set_title(
        f"Query Image\n{image_name}",
        fontsize=13,
    )

    ax_image.axis("off")

    # --------------------------------------------------------
    # Retrieved captions
    # --------------------------------------------------------

    ax_text = fig.add_subplot(
        grid[0, 1]
    )

    ax_text.axis("off")

    y_position = 0.95

    for rank, (
        score,
        caption_index_tensor,
    ) in enumerate(
        zip(scores, indices),
        start=1,
    ):

        caption_index = (
            caption_index_tensor.item()
        )

        caption = captions[
            caption_index
        ]

        is_correct = (
            caption_index
            in correct_indices
        )

        if is_correct:

            mark = "✓"

        else:

            mark = "✗"

        text = (
            f"{rank}. {mark} "
            f"[{score.item():.3f}]\n"
            f"{caption}"
        )

        ax_text.text(
            0.02,
            y_position,
            text,
            fontsize=10.5,
            va="top",
            wrap=True,
        )

        y_position -= 0.18

    ax_text.set_title(
        f"Top-{TOP_K} Captions\n"
        f"{num_correct}/{TOP_K} Ground-Truth Captions",
        fontsize=13,
    )

    # --------------------------------------------------------
    # Overall title
    # --------------------------------------------------------

    fig.suptitle(
        "Image → Text Retrieval",
        fontsize=17,
        y=0.98,
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    output_dir = (
        OUTPUT_DIR
        / "image_to_text"
        / category
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / f"image_{image_index}.png"
    )

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate qualitative CLIP "
            "retrieval examples."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "text-to-image",
            "image-to-text",
            "both",
        ],
        default="both",
        help="Retrieval direction.",
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=1,
        help=(
            "Number of examples per "
            "category."
        ),
    )

    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help=(
            "Generate a specific query "
            "instead of automatic examples."
        ),
    )

    parser.add_argument(
        "--recompute",
        action="store_true",
        help=(
            "Recompute embeddings instead "
            "of using cached embeddings."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print("=" * 70)
    print("Qualitative CLIP Retrieval")
    print("=" * 70)

    print(
        f"\nModel: {MODEL_PATH}"
    )

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

    print(
        "\nLoading CLIP..."
    )

    processor = (
        AutoProcessor.from_pretrained(
            MODEL_PATH
        )
    )

    model = (
        CLIPModel.from_pretrained(
            MODEL_PATH
        )
    )

    model = model.to(device)
    model.eval()

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    print(
        "\nLoading test data..."
    )

    (
        df,
        image_names,
        captions,
    ) = load_test_data()

    print(
        f"Test images: "
        f"{len(image_names)}"
    )

    print(
        f"Test captions: "
        f"{len(captions)}"
    )

    # --------------------------------------------------------
    # Load embeddings
    # --------------------------------------------------------

    (
        image_embeddings,
        text_embeddings,
    ) = get_embeddings(
        model=model,
        processor=processor,
        image_names=image_names,
        captions=captions,
        device=device,
        recompute=args.recompute,
    )

    # --------------------------------------------------------
    # Specific index
    # --------------------------------------------------------

    if args.index is not None:

        if args.mode == "text-to-image":

            if not (
                0
                <= args.index
                < len(captions)
            ):
                raise IndexError(
                    "Caption index is out of range."
                )

            visualize_text_to_image(
                caption_index=args.index,
                captions=captions,
                image_names=image_names,
                text_embeddings=text_embeddings,
                image_embeddings=image_embeddings,
                category="manual",
            )

        elif args.mode == "image-to-text":

            if not (
                0
                <= args.index
                < len(image_names)
            ):
                raise IndexError(
                    "Image index is out of range."
                )

            visualize_image_to_text(
                image_index=args.index,
                image_names=image_names,
                captions=captions,
                image_embeddings=image_embeddings,
                text_embeddings=text_embeddings,
                category="manual",
            )

        else:

            raise ValueError(
                "--index requires either "
                "--mode text-to-image or "
                "--mode image-to-text."
            )

        return

    # --------------------------------------------------------
    # Automatic examples
    # --------------------------------------------------------

    if args.mode in (
        "text-to-image",
        "both",
    ):

        print(
            "\nFinding Text → Image examples..."
        )

        text_examples = (
            find_text_to_image_examples(
                text_embeddings,
                image_embeddings,
                args.examples,
            )
        )

        for category, indices in (
            text_examples.items()
        ):

            print(
                f"\n{category.upper()}: "
                f"{indices}"
            )

            for caption_index in indices:

                visualize_text_to_image(
                    caption_index=caption_index,
                    captions=captions,
                    image_names=image_names,
                    text_embeddings=text_embeddings,
                    image_embeddings=image_embeddings,
                    category=category,
                )

    if args.mode in (
        "image-to-text",
        "both",
    ):

        print(
            "\nFinding Image → Text examples..."
        )

        image_examples = (
            find_image_to_text_examples(
                image_embeddings,
                text_embeddings,
                args.examples,
            )
        )

        for category, indices in (
            image_examples.items()
        ):

            print(
                f"\n{category.upper()}: "
                f"{indices}"
            )

            for image_index in indices:

                visualize_image_to_text(
                    image_index=image_index,
                    image_names=image_names,
                    captions=captions,
                    image_embeddings=image_embeddings,
                    text_embeddings=text_embeddings,
                    category=category,
                )

    print(
        "\n" + "=" * 70
    )

    print(
        "Qualitative analysis complete!"
    )

    print(
        f"Results saved to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()