from pathlib import Path
import random

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class Flickr30KTrainDataset(Dataset):
    """
    Flickr30K dataset for CLIP fine-tuning.

    Each image has five captions. For every sample,
    one caption is selected from the five captions.

    This ensures that each image appears exactly once
    in an epoch while allowing different captions to
    be selected across epochs.
    """

    def __init__(
        self,
        csv_file,
        image_dir,
        transform=None,
        seed=42,
    ):
        self.csv_file = Path(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.seed = seed

        # ----------------------------------------------------
        # Load CSV
        # ----------------------------------------------------

        self.data = pd.read_csv(self.csv_file)

        if "image" not in self.data.columns:
            raise ValueError(
                "CSV file must contain an 'image' column."
            )

        if "caption" not in self.data.columns:
            raise ValueError(
                "CSV file must contain a 'caption' column."
            )

        # ----------------------------------------------------
        # Validate image names
        # ----------------------------------------------------

        if self.data["image"].isna().any():
            raise ValueError(
                "Dataset contains missing image names."
            )

        # Convert image names to strings
        self.data["image"] = (
            self.data["image"]
            .astype(str)
            .str.strip()
        )

        if (self.data["image"] == "").any():
            raise ValueError(
                "Dataset contains empty image names."
            )

        # ----------------------------------------------------
        # Validate captions
        # ----------------------------------------------------

        missing_captions = self.data["caption"].isna()

        if missing_captions.any():
            num_missing = missing_captions.sum()

            raise ValueError(
                f"Dataset contains {num_missing} missing "
                f"captions."
            )

        # Convert captions to strings and remove whitespace
        self.data["caption"] = (
            self.data["caption"]
            .astype(str)
            .str.strip()
        )

        # Check for empty captions
        empty_captions = (
            self.data["caption"] == ""
        )

        if empty_captions.any():
            num_empty = empty_captions.sum()

            raise ValueError(
                f"Dataset contains {num_empty} empty "
                f"captions."
            )

        # ----------------------------------------------------
        # Group captions by image
        # ----------------------------------------------------

        self.image_captions = (
            self.data
            .groupby("image")["caption"]
            .apply(list)
            .to_dict()
        )

        self.image_names = list(
            self.image_captions.keys()
        )

        # ----------------------------------------------------
        # Verify exactly 5 captions per image
        # ----------------------------------------------------

        invalid = {
            image: len(captions)
            for image, captions
            in self.image_captions.items()
            if len(captions) != 5
        }

        if invalid:
            raise ValueError(
                f"{len(invalid)} images do not have "
                f"exactly 5 captions."
            )

        # ----------------------------------------------------
        # Verify all captions are strings
        # ----------------------------------------------------

        invalid_caption_types = []

        for image, captions in self.image_captions.items():
            for caption in captions:
                if not isinstance(caption, str):
                    invalid_caption_types.append(
                        (image, type(caption))
                    )

        if invalid_caption_types:
            raise TypeError(
                "Dataset contains captions that are not "
                "strings."
            )

        # ----------------------------------------------------
        # Verify image files
        # ----------------------------------------------------

        missing_images = []

        for image_name in self.image_names:
            image_path = self.image_dir / image_name

            if not image_path.exists():
                missing_images.append(image_name)

        if missing_images:
            raise FileNotFoundError(
                f"{len(missing_images)} image files were "
                f"not found."
            )

        # ----------------------------------------------------
        # Random generator for reproducibility
        # ----------------------------------------------------

        self.rng = random.Random(seed)

        print(
            f"Loaded {len(self.image_names)} images "
            f"with {len(self.data)} captions."
        )

    def __len__(self):
        """
        Number of samples.

        One sample = one image.
        """

        return len(self.image_names)

    def __getitem__(self, index):
        """
        Return one image and one randomly selected caption.
        """

        image_name = self.image_names[index]

        captions = self.image_captions[
            image_name
        ]

        # Select one of the five captions
        caption = self.rng.choice(captions)

        # Final safety check
        if not isinstance(caption, str):
            raise TypeError(
                f"Invalid caption type for "
                f"{image_name}: {type(caption)}"
            )

        if not caption.strip():
            raise ValueError(
                f"Empty caption for {image_name}"
            )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image_path = (
            self.image_dir / image_name
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # Optional transform
        # ----------------------------------------------------

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "caption": caption,
            "image_name": image_name,
        }

    def set_seed(self, seed):
        """
        Reset the caption sampling seed.
        """

        self.seed = seed
        self.rng = random.Random(seed)