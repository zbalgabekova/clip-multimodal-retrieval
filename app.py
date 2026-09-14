"""
Gradio demo for the fine-tuned CLIP Flickr30K retrieval project.

Features
--------
1. Text -> Image retrieval
2. Image -> Text retrieval

The app uses:
- checkpoints/best              : fine-tuned CLIP
- data/processed/embeddings/finetuned/
    test_image_embeddings.pt     : test image embeddings
    test_text_embeddings.pt      : test caption embeddings
- data/processed/test.csv        : Flickr30K test captions

Run from the Project3 root:
    python app.py

If Gradio is not installed:
    pip install gradio
"""

from pathlib import Path

import gradio as gr
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, AutoProcessor


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_DIR = ROOT / "checkpoints" / "best"

TEST_CSV = ROOT / "data" / "processed" / "test.csv"

IMAGE_EMBEDDINGS_FILE = (
    ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "finetuned"
    / "test_image_embeddings.pt"
)

TEXT_EMBEDDINGS_FILE = (
    ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "finetuned"
    / "test_text_embeddings.pt"
)

IMAGE_DIR = (
    ROOT
    / "data"
    / "raw"
    / "Images"
)


# ============================================================
# Settings
# ============================================================

TOP_K = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Load model and data once
# ============================================================

print("=" * 70)
print("CLIP Multimodal Retrieval Demo")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Model: {MODEL_DIR}")


if not MODEL_DIR.exists():
    raise FileNotFoundError(
        f"Fine-tuned model not found:\n{MODEL_DIR}\n\n"
        "Train the model first and make sure checkpoints/final exists."
    )

if not TEST_CSV.exists():
    raise FileNotFoundError(
        f"Test CSV not found:\n{TEST_CSV}"
    )

if not IMAGE_EMBEDDINGS_FILE.exists():
    raise FileNotFoundError(
        f"Image embeddings not found:\n{IMAGE_EMBEDDINGS_FILE}\n\n"
        "Run evaluation/retrieval.py with the fine-tuned model first."
    )

if not TEXT_EMBEDDINGS_FILE.exists():
    raise FileNotFoundError(
        f"Text embeddings not found:\n{TEXT_EMBEDDINGS_FILE}\n\n"
        "Run evaluation/retrieval.py with the fine-tuned model first."
    )

if not IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"Flickr30K image directory not found:\n{IMAGE_DIR}"
    )


print("\nLoading fine-tuned CLIP...")

processor = AutoProcessor.from_pretrained(MODEL_DIR)

model = CLIPModel.from_pretrained(MODEL_DIR)
model = model.to(DEVICE)
model.eval()

print("CLIP loaded successfully.")


print("\nLoading test data...")

df = pd.read_csv(TEST_CSV)

required_columns = {"image", "caption"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"test.csv must contain columns: {required_columns}"
    )

# Remove accidental NaN captions.
df = df.dropna(subset=["image", "caption"]).reset_index(drop=True)

# Keep the first occurrence of every test image.
image_names = df["image"].drop_duplicates().tolist()

captions = df["caption"].tolist()


print(f"Test images: {len(image_names)}")
print(f"Test captions: {len(captions)}")


print("\nLoading precomputed embeddings...")

test_image_embeddings = torch.load(
    IMAGE_EMBEDDINGS_FILE,
    map_location="cpu",
    weights_only=True,
).float()

test_text_embeddings = torch.load(
    TEXT_EMBEDDINGS_FILE,
    map_location="cpu",
    weights_only=True,
).float()

test_image_embeddings = F.normalize(
    test_image_embeddings,
    p=2,
    dim=1,
)

test_text_embeddings = F.normalize(
    test_text_embeddings,
    p=2,
    dim=1,
)


if len(image_names) != len(test_image_embeddings):
    raise ValueError(
        "Number of test images does not match image embeddings."
    )

if len(captions) != len(test_text_embeddings):
    raise ValueError(
        "Number of test captions does not match text embeddings."
    )


print("Embeddings loaded successfully.")

print("=" * 70)


# ============================================================
# CLIP feature extraction
# ============================================================

@torch.no_grad()
def encode_text(text):
    """Encode a text query into the CLIP shared embedding space."""

    inputs = processor(
        text=[text],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    text_outputs = model.text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    text_features = model.text_projection(
        text_outputs.pooler_output
    )

    text_features = F.normalize(
        text_features,
        p=2,
        dim=-1,
    )

    return text_features.cpu()


@torch.no_grad()
def encode_image(image):
    """Encode a PIL image into the CLIP shared embedding space."""

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(DEVICE)

    vision_outputs = model.vision_model(
        pixel_values=pixel_values
    )

    image_features = model.visual_projection(
        vision_outputs.pooler_output
    )

    image_features = F.normalize(
        image_features,
        p=2,
        dim=-1,
    )

    return image_features.cpu()


# ============================================================
# Helpers
# ============================================================

def get_image_path(image_name):
    """Return the local Flickr30K image path."""

    path = IMAGE_DIR / str(image_name)

    if not path.exists():
        return None

    return str(path)


def format_score(score):
    return f"{float(score):.4f}"


# ============================================================
# Text -> Image
# ============================================================

def text_to_image(query, top_k):
    """
    Retrieve the most similar Flickr30K test images for a text query.
    """

    if query is None or not query.strip():
        return [], "Please enter a text query."

    top_k = int(top_k)

    query_embedding = encode_text(
        query.strip()
    )

    similarities = (
        query_embedding
        @ test_image_embeddings.T
    )[0]

    values, indices = torch.topk(
        similarities,
        k=min(top_k, len(image_names)),
    )

    results = []

    for rank, (score, index) in enumerate(
        zip(values, indices),
        start=1,
    ):
        index = int(index)
        score = float(score)

        image_path = get_image_path(
            image_names[index]
        )

        if image_path is None:
            continue

        results.append(
            (
                image_path,
                f"Rank {rank} | Similarity: {format_score(score)}"
            )
        )

    info = (
        f"**Query:** {query.strip()}  \n"
        f"**Results:** {len(results)}"
    )

    return results, info


# ============================================================
# Image -> Text
# ============================================================

def image_to_text(image, top_k):
    """
    Retrieve the most similar Flickr30K captions for an uploaded image.
    """

    if image is None:
        return [], "Please upload an image."

    top_k = int(top_k)

    # Gradio can provide a NumPy array depending on the version/config.
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")

    image_embedding = encode_image(
        image
    )

    similarities = (
        image_embedding
        @ test_text_embeddings.T
    )[0]

    values, indices = torch.topk(
        similarities,
        k=min(top_k, len(captions)),
    )

    results = []

    for rank, (score, index) in enumerate(
        zip(values, indices),
        start=1,
    ):
        index = int(index)
        score = float(score)

        caption = str(
            captions[index]
        ).strip()

        results.append(
            f"**{rank}.** [{format_score(score)}] {caption}"
        )

    info = (
        f"**Top-{len(results)} retrieved captions**"
    )

    return "\n\n".join(results), info


# ============================================================
# Gradio UI
# ============================================================

DESCRIPTION = """
This demo uses a **fine-tuned CLIP model trained on Flickr30K**
to perform cross-modal retrieval.

- **Text → Image:** enter a natural-language description and retrieve similar images.
- **Image → Text:** upload an image and retrieve the most similar captions.

The retrieval database contains the Flickr30K test images and captions.
"""

with gr.Blocks(
    title="CLIP Multimodal Retrieval"
) as demo:

    gr.Markdown(
        "# CLIP Multimodal Retrieval"
    )

    gr.Markdown(
        DESCRIPTION
    )

    gr.Markdown(
        f"**Model:** Fine-tuned CLIP ViT-B/32  |  "
        f"**Device:** `{DEVICE}`  |  "
        f"**Top-K:** {TOP_K}"
    )

    with gr.Tabs():

        # ----------------------------------------------------
        # Text -> Image
        # ----------------------------------------------------

        with gr.Tab(
            "Text → Image"
        ):

            gr.Markdown(
                "### Search images using natural language"
            )

            text_query = gr.Textbox(
                label="Text query",
                placeholder=(
                    "Example: a woman running a marathon"
                ),
                lines=2,
            )

            text_top_k = gr.Slider(
                minimum=1,
                maximum=10,
                value=TOP_K,
                step=1,
                label="Number of results",
            )

            text_button = gr.Button(
                "Search Images",
                variant="primary",
            )

            text_status = gr.Markdown()

            text_gallery = gr.Gallery(
                label="Retrieved Images",
                columns=5,
                rows=2,
                height="auto",
                object_fit="contain",
                preview=True,
            )

            text_button.click(
                fn=text_to_image,
                inputs=[
                    text_query,
                    text_top_k,
                ],
                outputs=[
                    text_gallery,
                    text_status,
                ],
            )

            text_query.submit(
                fn=text_to_image,
                inputs=[
                    text_query,
                    text_top_k,
                ],
                outputs=[
                    text_gallery,
                    text_status,
                ],
            )

        # ----------------------------------------------------
        # Image -> Text
        # ----------------------------------------------------

        with gr.Tab(
            "Image → Text"
        ):

            gr.Markdown(
                "### Search captions using an image"
            )

            image_input = gr.Image(
                label="Upload an image",
                type="pil",
            )

            image_top_k = gr.Slider(
                minimum=1,
                maximum=10,
                value=TOP_K,
                step=1,
                label="Number of results",
            )

            image_button = gr.Button(
                "Search Captions",
                variant="primary",
            )

            image_status = gr.Markdown()

            image_results = gr.Markdown(
                label="Retrieved Captions"
            )

            image_button.click(
                fn=image_to_text,
                inputs=[
                    image_input,
                    image_top_k,
                ],
                outputs=[
                    image_results,
                    image_status,
                ],
            )


if __name__ == "__main__":
    demo.launch()
