from pathlib import Path
import random
import csv


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

CAPTIONS_FILE = DATA_DIR / "captions.txt"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 42


# ============================================================
# Load captions
# ============================================================

def load_captions():
    captions = {}

    with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            # Skip empty rows
            if not row:
                continue

            # Skip header
            if row[0].strip().lower() == "image":
                continue

            if len(row) < 2:
                continue

            image_name = row[0].strip()
            caption = ",".join(row[1:]).strip()

            if image_name not in captions:
                captions[image_name] = []

            captions[image_name].append(caption)

    return captions


# ============================================================
# Split images
# ============================================================

def create_splits(image_names):
    random.seed(SEED)

    image_names = list(image_names)
    random.shuffle(image_names)

    total = len(image_names)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = image_names[:train_end]
    val_images = image_names[train_end:val_end]
    test_images = image_names[val_end:]

    return train_images, val_images, test_images


# ============================================================
# Save split
# ============================================================

def save_split(image_names, captions, output_file):

    with open(output_file, "w", encoding="utf-8", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["image", "caption"])

        for image_name in image_names:

            for caption in captions[image_name]:

                writer.writerow([
                    image_name,
                    caption
                ])


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Flickr30K Dataset Preparation")
    print("=" * 60)

    # Load captions
    captions = load_captions()

    print(f"\nImages with captions: {len(captions)}")

    # Verify exactly 5 captions per image
    caption_counts = {
        image: len(caps)
        for image, caps in captions.items()
    }

    invalid = {
        image: count
        for image, count in caption_counts.items()
        if count != 5
    }

    if invalid:
        print("\nWARNING:")
        print(f"{len(invalid)} images do not have exactly 5 captions.")

        for image, count in list(invalid.items())[:10]:
            print(f"  {image}: {count} captions")

    else:
        print("All images have exactly 5 captions.")

    # Create splits
    train_images, val_images, test_images = create_splits(
        captions.keys()
    )

    print("\nSplit sizes:")
    print(f"  Train:      {len(train_images)} images")
    print(f"  Validation: {len(val_images)} images")
    print(f"  Test:       {len(test_images)} images")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save splits
    save_split(
        train_images,
        captions,
        OUTPUT_DIR / "train.csv"
    )

    save_split(
        val_images,
        captions,
        OUTPUT_DIR / "val.csv"
    )

    save_split(
        test_images,
        captions,
        OUTPUT_DIR / "test.csv"
    )

    # Verify no image overlap
    train_set = set(train_images)
    val_set = set(val_images)
    test_set = set(test_images)

    print("\nOverlap checks:")

    print(
        f"  Train / Val:  "
        f"{len(train_set & val_set)}"
    )

    print(
        f"  Train / Test: "
        f"{len(train_set & test_set)}"
    )

    print(
        f"  Val / Test:   "
        f"{len(val_set & test_set)}"
    )

    print("\nFiles created:")

    print(f"  {OUTPUT_DIR / 'train.csv'}")
    print(f"  {OUTPUT_DIR / 'val.csv'}")
    print(f"  {OUTPUT_DIR / 'test.csv'}")

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()