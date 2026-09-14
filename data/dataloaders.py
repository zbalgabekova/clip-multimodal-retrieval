from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from data.dataset import Flickr30KDataset


ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "data" / "raw" / "Images"
PROCESSED_DIR = ROOT / "data" / "processed"


def get_transform():
    """
    Basic image transformation.

    CLIP-specific preprocessing will be added later
    when we load the CLIP processor.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def create_dataloader(
    split,
    batch_size=32,
    shuffle=False,
    num_workers=0,
):
    """
    Create a DataLoader for a Flickr30K split.

    Args:
        split: 'train', 'val', or 'test'
        batch_size: number of samples per batch
        shuffle: whether to shuffle the dataset
        num_workers: number of DataLoader workers
    """

    if split not in {"train", "val", "test"}:
        raise ValueError(
            "split must be 'train', 'val', or 'test'"
        )

    csv_file = PROCESSED_DIR / f"{split}.csv"

    dataset = Flickr30KDataset(
        csv_file=csv_file,
        image_dir=IMAGE_DIR,
        transform=get_transform(),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return dataloader