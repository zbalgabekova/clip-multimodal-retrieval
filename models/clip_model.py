import torch
from transformers import CLIPModel, AutoProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_clip():
    device = get_device()

    print(f"Loading CLIP: {MODEL_NAME}")
    print(f"Device: {device}")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    model = CLIPModel.from_pretrained(MODEL_NAME)

    model = model.to(device)
    model.eval()

    return model, processor, device