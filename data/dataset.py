from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class Flickr30KDataset(Dataset):
    """
    PyTorch Dataset for Flickr30K image-caption pairs.

    Each image has five captions, so an image appears
    five times in the dataframe.
    """

    def __init__(
        self,
        csv_file,
        image_dir,
        transform=None,
    ):
        self.csv_file = Path(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform

        self.data = pd.read_csv(self.csv_file)

        if "image" not in self.data.columns:
            raise ValueError(
                "CSV file must contain an 'image' column."
            )

        if "caption" not in self.data.columns:
            raise ValueError(
                "CSV file must contain a 'caption' column."
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_name = row["image"]
        caption = row["caption"]

        image_path = self.image_dir / image_name

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "caption": caption,
            "image_name": image_name,
        }