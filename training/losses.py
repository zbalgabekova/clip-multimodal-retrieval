import torch
import torch.nn.functional as F


def clip_contrastive_loss(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    logit_scale: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """
    Compute the symmetric CLIP contrastive loss.

    Each image at index i is assumed to match the text
    at index i.

    Args:
        image_embeddings:
            Image embeddings of shape [batch_size, embedding_dim].

        text_embeddings:
            Text embeddings of shape [batch_size, embedding_dim].

        logit_scale:
            Temperature scaling factor applied to the
            image-text similarity matrix.

            CLIP normally learns this parameter. The value
            passed here should already be positive.

    Returns:
        Scalar symmetric contrastive loss.
    """

    if image_embeddings.ndim != 2:
        raise ValueError(
            "image_embeddings must be a 2D tensor "
            "[batch_size, embedding_dim]."
        )

    if text_embeddings.ndim != 2:
        raise ValueError(
            "text_embeddings must be a 2D tensor "
            "[batch_size, embedding_dim]."
        )

    if image_embeddings.shape[0] != text_embeddings.shape[0]:
        raise ValueError(
            "Image and text batches must have the same size."
        )

    if image_embeddings.shape[1] != text_embeddings.shape[1]:
        raise ValueError(
            "Image and text embeddings must have the same "
            "embedding dimension."
        )

    batch_size = image_embeddings.shape[0]

    if batch_size < 2:
        raise ValueError(
            "Batch size must be at least 2 for contrastive learning."
        )

    # Normalize embeddings.
    image_embeddings = F.normalize(
        image_embeddings,
        p=2,
        dim=-1,
    )

    text_embeddings = F.normalize(
        text_embeddings,
        p=2,
        dim=-1,
    )

    # --------------------------------------------------------
    # Image-text similarity matrix
    # --------------------------------------------------------

    logits_per_image = (
        image_embeddings @ text_embeddings.T
    )

    # Convert logit_scale to a tensor on the same device.
    if not isinstance(logit_scale, torch.Tensor):
        logit_scale = torch.tensor(
            logit_scale,
            device=image_embeddings.device,
            dtype=image_embeddings.dtype,
        )
    else:
        logit_scale = logit_scale.to(
            device=image_embeddings.device,
            dtype=image_embeddings.dtype,
        )

    logits_per_image = (
        logits_per_image * logit_scale
    )

    # The transpose gives Text → Image similarities.
    logits_per_text = logits_per_image.T

    # --------------------------------------------------------
    # Correct labels
    # --------------------------------------------------------

    labels = torch.arange(
        batch_size,
        device=image_embeddings.device,
    )

    # --------------------------------------------------------
    # Symmetric loss
    # --------------------------------------------------------

    image_to_text_loss = F.cross_entropy(
        logits_per_image,
        labels,
    )

    text_to_image_loss = F.cross_entropy(
        logits_per_text,
        labels,
    )

    loss = (
        image_to_text_loss
        + text_to_image_loss
    ) / 2.0

    return loss