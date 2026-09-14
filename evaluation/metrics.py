import torch


def recall_at_k(
    similarity: torch.Tensor,
    ground_truth,
    k: int,
) -> float:
    """
    Calculate Recall@K for image-text retrieval.

    Args:
        similarity:
            2D tensor of shape [num_queries, num_candidates].
            Higher values indicate greater similarity.

        ground_truth:
            Ground-truth candidate indices for each query.

            Can be:
                - list[int]
                - list[list[int]]
                - torch.Tensor

            Examples:

            Text -> Image:
                [5, 12, 7, 20]

            Image -> Text:
                [[0, 1, 2, 3, 4],
                 [5, 6, 7, 8, 9],
                 ...]

        k:
            Number of retrieved candidates.

    Returns:
        Recall@K as a percentage in the range [0, 100].
    """

    if similarity.ndim != 2:
        raise ValueError(
            "similarity must be a 2D tensor "
            "[num_queries, num_candidates]."
        )

    num_queries, num_candidates = similarity.shape

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    k = min(k, num_candidates)

    # Convert ground truth to a list of lists.
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.tolist()

    if not ground_truth:
        raise ValueError("ground_truth cannot be empty.")

    if isinstance(ground_truth[0], int):
        ground_truth = [
            [index]
            for index in ground_truth
        ]

    if len(ground_truth) != num_queries:
        raise ValueError(
            "Number of ground-truth entries must match "
            "the number of queries."
        )

    # Get indices of top-k candidates.
    top_k = torch.topk(
        similarity,
        k=k,
        dim=1,
    ).indices

    correct = 0

    for query_idx in range(num_queries):

        retrieved = set(
            top_k[query_idx].tolist()
        )

        targets = set(
            ground_truth[query_idx]
        )

        # A query is correct if ANY ground-truth
        # candidate appears in the top-k results.
        if retrieved.intersection(targets):
            correct += 1

    recall = correct / num_queries * 100.0

    return recall


def retrieval_metrics(
    similarity: torch.Tensor,
    ground_truth,
    ks=(1, 5, 10),
) -> dict:
    """
    Calculate multiple Recall@K values.

    Returns:
        Dictionary such as:

        {
            "R@1": 42.31,
            "R@5": 68.72,
            "R@10": 78.54
        }
    """

    results = {}

    for k in ks:
        results[f"R@{k}"] = recall_at_k(
            similarity=similarity,
            ground_truth=ground_truth,
            k=k,
        )

    return results