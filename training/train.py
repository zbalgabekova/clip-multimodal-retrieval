from pathlib import Path
import json
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import CLIPModel, AutoProcessor
from tqdm import tqdm

from training.train_dataset import Flickr30KTrainDataset
from training.losses import clip_contrastive_loss


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TRAIN_CSV = ROOT / "data" / "processed" / "train.csv"
VAL_CSV = ROOT / "data" / "processed" / "val.csv"

IMAGE_DIR = ROOT / "data" / "raw" / "Images"

OUTPUT_DIR = ROOT / "checkpoints"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_NAME = "openai/clip-vit-base-patch32"


# ============================================================
# Training settings
# ============================================================

# Keep 5000 for this experiment.
# Set to None later for the full training set.
MAX_IMAGES = 5000

# Physical batch size.
BATCH_SIZE = 4

# Validation batch size.
VAL_BATCH_SIZE = 4

# Gradient accumulation.
#
# 4 physical samples x 8 accumulation steps
# = effective optimizer batch size of 32.
#
# IMPORTANT:
# This does NOT create 32 in-batch negatives
# for the CLIP contrastive loss.
GRADIENT_ACCUMULATION_STEPS = 8

NUM_WORKERS = 0

EPOCHS = 3

LEARNING_RATE = 1e-6
WEIGHT_DECAY = 0.01

MAX_GRAD_NORM = 1.0

SEED = 42


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Device
# ============================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# Collate function
# ============================================================

def collate_fn(batch):
    """
    Keep images as PIL images and captions as strings.
    CLIP processor handles preprocessing later.
    """

    images = [
        item["image"]
        for item in batch
    ]

    captions = [
        item["caption"]
        for item in batch
    ]

    image_names = [
        item["image_name"]
        for item in batch
    ]

    return {
        "image": images,
        "caption": captions,
        "image_name": image_names,
    }


# ============================================================
# DataLoaders
# ============================================================

def create_train_dataloader():

    dataset = Flickr30KTrainDataset(
        csv_file=TRAIN_CSV,
        image_dir=IMAGE_DIR,
        seed=SEED,
    )

    # --------------------------------------------------------
    # Limit training dataset
    # --------------------------------------------------------

    if MAX_IMAGES is not None:

        max_images = min(
            MAX_IMAGES,
            len(dataset),
        )

        dataset = Subset(
            dataset,
            range(max_images),
        )

    print(
        f"Using {len(dataset)} images for training"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    return dataset, dataloader


def create_val_dataloader():

    dataset = Flickr30KTrainDataset(
        csv_file=VAL_CSV,
        image_dir=IMAGE_DIR,
        seed=SEED,
    )

    print(
        f"Using {len(dataset)} images for validation"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    return dataset, dataloader


# ============================================================
# CLIP embeddings
# ============================================================

def get_image_features(
    model,
    pixel_values,
):
    """
    Extract projected and normalized CLIP image embeddings.
    """

    vision_outputs = model.vision_model(
        pixel_values=pixel_values
    )

    image_features = (
        vision_outputs.pooler_output
    )

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
    Extract projected and normalized CLIP text embeddings.
    """

    text_outputs = model.text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    text_features = (
        text_outputs.pooler_output
    )

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
# Prepare batch
# ============================================================

def prepare_batch(
    batch,
    processor,
    device,
):
    """
    Convert images and captions into CLIP inputs.
    """

    images = batch["image"]

    captions = [
        str(caption).strip()
        for caption in batch["caption"]
    ]

    # Safety check
    if not all(captions):
        raise ValueError(
            f"Empty caption found: {captions}"
        )

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    image_inputs = processor(
        images=images,
        return_tensors="pt",
    )

    pixel_values = image_inputs[
        "pixel_values"
    ].to(
        device,
        non_blocking=True,
    )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    text_inputs = processor(
        text=captions,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = text_inputs[
        "input_ids"
    ].to(device)

    attention_mask = text_inputs[
        "attention_mask"
    ].to(device)

    return (
        pixel_values,
        input_ids,
        attention_mask,
    )


# ============================================================
# Calculate loss
# ============================================================

def calculate_batch_loss(
    model,
    processor,
    batch,
    device,
):
    """
    Calculate CLIP contrastive loss for one batch.
    """

    (
        pixel_values,
        input_ids,
        attention_mask,
    ) = prepare_batch(
        batch=batch,
        processor=processor,
        device=device,
    )

    # --------------------------------------------------------
    # Image embeddings
    # --------------------------------------------------------

    image_features = get_image_features(
        model=model,
        pixel_values=pixel_values,
    )

    # --------------------------------------------------------
    # Text embeddings
    # --------------------------------------------------------

    text_features = get_text_features(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    # --------------------------------------------------------
    # CLIP temperature
    # --------------------------------------------------------

    logit_scale = model.logit_scale.exp()

    # --------------------------------------------------------
    # Contrastive loss
    # --------------------------------------------------------

    loss = clip_contrastive_loss(
        image_embeddings=image_features,
        text_embeddings=text_features,
        logit_scale=logit_scale,
    )

    return loss


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(
    model,
    processor,
    dataloader,
    optimizer,
    device,
    epoch,
):
    """
    Train CLIP for one epoch using gradient accumulation.
    """

    model.train()

    total_loss = 0.0

    num_batches = len(dataloader)

    optimizer.zero_grad(
        set_to_none=True
    )

    progress = tqdm(
        enumerate(dataloader),
        total=num_batches,
        desc=f"Epoch {epoch}",
    )

    for batch_index, batch in progress:

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        loss = calculate_batch_loss(
            model=model,
            processor=processor,
            batch=batch,
            device=device,
        )

        # ----------------------------------------------------
        # Keep original loss for logging
        # ----------------------------------------------------

        total_loss += loss.item()

        # ----------------------------------------------------
        # Scale loss before accumulation
        # ----------------------------------------------------

        scaled_loss = (
            loss
            / GRADIENT_ACCUMULATION_STEPS
        )

        scaled_loss.backward()

        # ----------------------------------------------------
        # Determine whether to update
        # ----------------------------------------------------

        is_accumulation_step = (
            (batch_index + 1)
            % GRADIENT_ACCUMULATION_STEPS
            == 0
        )

        is_last_batch = (
            batch_index + 1
            == num_batches
        )

        if (
            is_accumulation_step
            or is_last_batch
        ):

            # ------------------------------------------------
            # Gradient clipping
            # ------------------------------------------------

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM,
            )

            # ------------------------------------------------
            # Update model
            # ------------------------------------------------

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        average_loss = (
            total_loss
            / (batch_index + 1)
        )

        progress.set_postfix(
            loss=f"{average_loss:.4f}"
        )

    return (
        total_loss
        / num_batches
    )


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    processor,
    dataloader,
    device,
    dataset,
):
    """
    Calculate validation contrastive loss.

    Validation does not use gradient accumulation.
    """

    model.eval()

    # Reset caption sampling so validation is deterministic.
    if hasattr(dataset, "set_seed"):
        dataset.set_seed(SEED)

    total_loss = 0.0

    progress = tqdm(
        dataloader,
        desc="Validation",
    )

    for batch in progress:

        loss = calculate_batch_loss(
            model=model,
            processor=processor,
            batch=batch,
            device=device,
        )

        total_loss += loss.item()

        average_loss = (
            total_loss
            / (progress.n + 1)
        )

        progress.set_postfix(
            val_loss=f"{average_loss:.4f}"
        )

    return (
        total_loss
        / len(dataloader)
    )


# ============================================================
# Save checkpoint
# ============================================================

def save_checkpoint(
    model,
    processor,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    filename,
):

    checkpoint_dir = (
        OUTPUT_DIR / filename
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model.save_pretrained(
        checkpoint_dir
    )

    processor.save_pretrained(
        checkpoint_dir
    )

    # --------------------------------------------------------
    # Save optimizer
    # --------------------------------------------------------

    torch.save(
        optimizer.state_dict(),
        checkpoint_dir / "optimizer.pt",
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "model_name": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,
        "effective_optimizer_batch_size":
            BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "max_images": MAX_IMAGES,
    }

    with open(
        checkpoint_dir / "training_info.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )

    print(
        f"\nCheckpoint saved to:\n"
        f"{checkpoint_dir}"
    )


# ============================================================
# Main
# ============================================================

def main():

    set_seed(SEED)

    device = get_device()

    print("=" * 60)
    print("CLIP Fine-Tuning")
    print("=" * 60)

    print(
        f"\nModel: {MODEL_NAME}"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Gradient accumulation: "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        f"Effective optimizer batch size: "
        f"{BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        f"Validation batch size: "
        f"{VAL_BATCH_SIZE}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"Max training images: "
        f"{MAX_IMAGES}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Load CLIP
    # --------------------------------------------------------

    print("\nLoading CLIP...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    model = CLIPModel.from_pretrained(
        MODEL_NAME
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Save parameter for update check
    # --------------------------------------------------------

    before = (
        model.visual_projection.weight
        .detach()
        .clone()
    )

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    print(
        "\nCreating training DataLoader..."
    )

    (
        train_dataset,
        train_dataloader,
    ) = create_train_dataloader()

    print(
        f"Training images: "
        f"{len(train_dataset)}"
    )

    print(
        f"Training batches: "
        f"{len(train_dataloader)}"
    )

    # --------------------------------------------------------
    # Validation data
    # --------------------------------------------------------

    print(
        "\nCreating validation DataLoader..."
    )

    (
        val_dataset,
        val_dataloader,
    ) = create_val_dataloader()

    print(
        f"Validation images: "
        f"{len(val_dataset)}"
    )

    print(
        f"Validation batches: "
        f"{len(val_dataloader)}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = []

    best_val_loss = float("inf")

    best_epoch = None

    start_time = time.time()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        epoch_start = time.time()

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        train_loss = train_one_epoch(
            model=model,
            processor=processor,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss = validate(
            model=model,
            processor=processor,
            dataloader=val_dataloader,
            device=device,
            dataset=val_dataset,
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        # ----------------------------------------------------
        # Epoch results
        # ----------------------------------------------------

        print(
            f"\nEpoch {epoch}/{EPOCHS}"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Val Loss:   {val_loss:.4f}"
        )

        print(
            f"Time: "
            f"{epoch_time / 60:.2f} min"
        )

        # ----------------------------------------------------
        # Store history
        # ----------------------------------------------------

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "time_seconds": epoch_time,
            }
        )

        # ----------------------------------------------------
        # Save epoch checkpoint
        # ----------------------------------------------------

        save_checkpoint(
            model=model,
            processor=processor,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            filename=f"epoch_{epoch}",
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_epoch = epoch

            print(
                "\nNew best model!"
            )

            print(
                f"Best validation loss: "
                f"{best_val_loss:.4f}"
            )

            save_checkpoint(
                model=model,
                processor=processor,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                filename="best",
            )

    # --------------------------------------------------------
    # Measure parameter change
    # --------------------------------------------------------

    after = (
        model.visual_projection.weight
        .detach()
    )

    change = (
        (after - before)
        .abs()
        .mean()
        .item()
    )

    print(
        f"\nMean parameter change: "
        f"{change:.10f}"
    )

    # --------------------------------------------------------
    # Total training time
    # --------------------------------------------------------

    total_time = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    final_train_loss = (
        history[-1]["train_loss"]
    )

    final_val_loss = (
        history[-1]["val_loss"]
    )

    save_checkpoint(
        model=model,
        processor=processor,
        optimizer=optimizer,
        epoch=EPOCHS,
        train_loss=final_train_loss,
        val_loss=final_val_loss,
        filename="final",
    )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_file = (
        OUTPUT_DIR
        / "training_history.json"
    )

    with open(
        history_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "Training complete!"
    )

    print(
        "=" * 60
    )

    print(
        f"\nTotal training time: "
        f"{total_time / 60:.2f} min"
    )

    print(
        f"Final train loss: "
        f"{final_train_loss:.4f}"
    )

    print(
        f"Final validation loss: "
        f"{final_val_loss:.4f}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"\nBest model:"
        f"\n{OUTPUT_DIR / 'best'}"
    )

    print(
        f"\nFinal model:"
        f"\n{OUTPUT_DIR / 'final'}"
    )


if __name__ == "__main__":
    main()