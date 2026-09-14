from pathlib import Path
from collections import Counter


DATA_DIR = Path("data/raw")
IMAGE_DIR = DATA_DIR / "Images"
CAPTIONS_FILE = DATA_DIR / "captions.txt"


def main():
    print("=" * 60)
    print("Flickr30K Dataset Inspection")
    print("=" * 60)

    # --------------------------------------------------
    # Images
    # --------------------------------------------------
    images = list(IMAGE_DIR.glob("*.jpg"))

    print(f"\nImages found: {len(images)}")

    # --------------------------------------------------
    # Captions
    # --------------------------------------------------
    with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Caption lines: {len(lines)}")

    # --------------------------------------------------
    # Parse captions
    # --------------------------------------------------
    image_names = []

    for line in lines:
        image_name = line.split(",", 1)[0]
        image_names.append(image_name)

    caption_counts = Counter(image_names)

    print(f"Unique images in captions: {len(caption_counts)}")

    # --------------------------------------------------
    # Caption distribution
    # --------------------------------------------------
    distribution = Counter(caption_counts.values())

    print("\nCaptions per image:")
    for num_captions, count in sorted(distribution.items()):
        print(f"  {num_captions}: {count} images")

    # --------------------------------------------------
    # Missing images
    # --------------------------------------------------
    image_set = {image.name for image in images}
    caption_image_set = set(caption_counts.keys())

    missing_images = caption_image_set - image_set
    images_without_captions = image_set - caption_image_set

    print(f"\nImages referenced by captions but missing: "
          f"{len(missing_images)}")

    print(f"Images without captions: "
          f"{len(images_without_captions)}")

    # --------------------------------------------------
    # Show examples
    # --------------------------------------------------
    print("\nExample captions:")

    shown = 0
    for line in lines:
        print(f"  {line}")

        shown += 1
        if shown >= 5:
            break

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()