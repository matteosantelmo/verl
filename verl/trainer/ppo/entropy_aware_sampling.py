from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from verl import DataProto
from verl.utils.torch_functional import masked_mean, masked_sum


def compute_sequence_entropy(
    entropies: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute sequence-level entropy by averaging token-level entropies.

    Args:
        entropies: Token-level entropies of shape (batch_size, response_length)
        response_mask: Mask for response tokens of shape (batch_size, response_length)

    Returns:
        Sequence-level entropy of shape (batch_size,)
    """
    # Compute masked mean of entropies for each sequence
    return masked_mean(entropies, response_mask, axis=-1)


def compute_sequence_log_prob(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute sequence-level log probability by summing token-level log probs.

    Args:
        log_probs: Token-level log probabilities of shape (batch_size, response_length)
        response_mask: Mask for response tokens of shape (batch_size, response_length)

    Returns:
        Sequence-level log probability of shape (batch_size,)
    """
    # Sum log probabilities over valid tokens
    return masked_sum(log_probs, response_mask, axis=-1)


def entropy_aware_subsample(
    batch: DataProto,
    target_n: int,
    selection_criterion: str,
    min_rollouts_for_sampling: int = 4,
) -> tuple[DataProto, dict]:
    """Subsample rollouts using entropy-aware selection for calibrated uncertainty.

    This function implements the entropy-aware sampling strategy:
    1. Groups rollouts by prompt (using uid)
    2. For each group, computes the proportion p of correct answers
    3. Selects p*n correct answers randomly
    4. Selects (1-p)*n incorrect answers with highest entropy (or log prob)

    Args:
        batch: DataProto containing:
            - batch["token_level_scores"]: Token-level scores (rewards)
            - batch["old_log_probs"]: Token-level log probabilities
            - batch["entropys"]: Token-level entropies (if selection_criterion="entropy")
            - batch["response_mask"]: Mask for response tokens
            - non_tensor_batch["uid"]: Unique identifier for each prompt
        target_n: Target number of rollouts per group (n in the description)
        selection_criterion: Either "entropy" or "log_prob"
        min_rollouts_for_sampling: Minimum rollouts required for sampling

    Returns:
        tuple:
            - Subsampled DataProto with n rollouts per group
            - Dictionary of metrics about the sampling process
    """
    if selection_criterion not in ["entropy", "log_prob"]:
        raise ValueError(f"Invalid selection_criterion: {selection_criterion}. Must be 'entropy' or 'log_prob'")

    # Get necessary tensors
    token_level_scores = batch.batch["token_level_scores"]
    response_mask = batch.batch["response_mask"]
    uids = batch.non_tensor_batch["uid"]

    # Compute sequence-level scores (rewards)
    sequence_scores = token_level_scores.sum(dim=-1)  # (batch_size,)

    # Compute sequence-level criterion values
    if selection_criterion == "entropy":
        if "entropys" not in batch.batch:
            raise ValueError("Entropy-aware sampling with criterion='entropy' requires 'entropys' in batch")
        criterion_values = compute_sequence_entropy(batch.batch["entropys"], response_mask)
    else:  # log_prob
        if "old_log_probs" not in batch.batch:
            raise ValueError("Entropy-aware sampling with criterion='log_prob' requires 'old_log_probs' in batch")
        criterion_values = compute_sequence_log_prob(batch.batch["old_log_probs"], response_mask)

    print(f"DEBUG: sequence_scores.shape = {sequence_scores.shape}")
    print(f"DEBUG: criterion_values.shape = {criterion_values.shape}")

    # Group rollouts by uid
    uid_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, uid in enumerate(uids):
        uid_to_indices[uid].append(i)

    # Select indices for each group
    selected_indices = []
    metrics = {
        "entropy_aware_sampling/groups_total": 0,
        "entropy_aware_sampling/avg_correct_proportion": 0.0,
        "entropy_aware_sampling/avg_rollouts_per_group": 0.0,
        "entropy_aware_sampling/groups_all_correct": 0,
        "entropy_aware_sampling/groups_all_incorrect": 0,
    }

    total_correct_proportion = 0.0
    total_rollouts = 0

    for uid, indices in uid_to_indices.items():
        metrics["entropy_aware_sampling/groups_total"] += 1
        group_size = len(indices)
        total_rollouts += group_size

        # Get scores and criterion values for this group
        group_scores = sequence_scores[indices]
        group_criterion = criterion_values[indices]

        # Separate correct (reward > 0) and incorrect (reward <= 0) indices
        # For binary rewards, correct means reward = 1, incorrect means reward = 0
        correct_mask = group_scores > 0
        correct_local_indices = [i for i, is_correct in enumerate(correct_mask.tolist()) if is_correct]
        incorrect_local_indices = [i for i, is_correct in enumerate(correct_mask.tolist()) if not is_correct]

        n_correct = len(correct_local_indices)
        n_incorrect = len(incorrect_local_indices)
        p = n_correct / group_size  # Proportion of correct answers
        total_correct_proportion += p

        # Track edge cases for metrics
        if n_correct == 0:
            metrics["entropy_aware_sampling/groups_all_incorrect"] += 1
        elif n_incorrect == 0:
            metrics["entropy_aware_sampling/groups_all_correct"] += 1

        # Number of correct and incorrect to select
        n_correct_to_select = int(round(p * target_n))
        n_incorrect_to_select = target_n - n_correct_to_select

        # Ensure we don't select more than available
        n_correct_to_select = min(n_correct_to_select, n_correct)
        n_incorrect_to_select = min(n_incorrect_to_select, n_incorrect)

        # Adjust if total is less than target_n
        total_to_select = n_correct_to_select + n_incorrect_to_select
        if total_to_select < target_n:
            # Fill up from whichever category has more available
            remaining = target_n - total_to_select
            extra_correct = min(remaining, n_correct - n_correct_to_select)
            n_correct_to_select += extra_correct
            remaining -= extra_correct
            extra_incorrect = min(remaining, n_incorrect - n_incorrect_to_select)
            n_incorrect_to_select += extra_incorrect

        # Select correct samples randomly
        np.random.shuffle(correct_local_indices)
        selected_correct = correct_local_indices[:n_correct_to_select]

        # Select incorrect samples based on criterion
        if selection_criterion == "entropy":
            # Select lowest entropy (most confident wrong answers)
            sorted_incorrect = sorted(
                incorrect_local_indices, key=lambda i: group_criterion[i].item(), reverse=False
            )
        else:  # log_prob
            # Select highest log prob (most confident wrong answers)
            sorted_incorrect = sorted(
                incorrect_local_indices, key=lambda i: group_criterion[i].item(), reverse=True
            )
        selected_incorrect = sorted_incorrect[:n_incorrect_to_select]

        selected_local = selected_correct + selected_incorrect
        selected_indices.extend([indices[i] for i in selected_local])

    # Compute average metrics
    n_groups = metrics["entropy_aware_sampling/groups_total"]
    if n_groups > 0:
        metrics["entropy_aware_sampling/avg_correct_proportion"] = total_correct_proportion / n_groups
        metrics["entropy_aware_sampling/avg_rollouts_per_group"] = total_rollouts / n_groups

    # Add batch size metrics
    metrics["entropy_aware_sampling/batch_size_before"] = len(uids)
    metrics["entropy_aware_sampling/batch_size_after"] = len(selected_indices)
    metrics["entropy_aware_sampling/target_n_per_group"] = target_n

    # Subsample the batch
    selected_indices = sorted(selected_indices)  # Keep order for consistency
    subsampled_batch = batch.select_idxs(torch.tensor(selected_indices))

    return subsampled_batch, metrics


def should_apply_entropy_aware_sampling(config) -> bool:
    """Check if entropy-aware sampling should be applied based on config.

    Args:
        config: The algorithm configuration object

    Returns:
        True if entropy-aware sampling should be applied
    """
    entropy_aware_config = config.get("entropy_aware_sampling", None)
    if entropy_aware_config is None:
        return False

    selection = entropy_aware_config.get("incorrect_rollouts_selection", None)
    return selection is not None and selection in ["entropy", "log_prob"]


def get_entropy_aware_sampling_config(config) -> tuple[Optional[str], float]:
    """Extract entropy-aware sampling configuration.

    Args:
        config: The algorithm configuration object

    Returns:
        tuple of (selection_criterion, over_sampling_ratio)
    """
    entropy_aware_config = config.get("entropy_aware_sampling", {})
    selection_criterion = entropy_aware_config.get("incorrect_rollouts_selection", None)
    over_sampling_ratio = entropy_aware_config.get("over_sampling_ratio", 2.0)

    return selection_criterion, over_sampling_ratio


def get_entropy_stats(batch: DataProto, prefix: str) -> dict:
    """
    Returns metrics about entropies and log-probabilities in the batch,
    with breakdowns for positive and negative rewards, as well as
    binned by group accuracy.
    """
    bins = torch.arange(0, 1, 0.25) + 0.25
    binned_entropies = {b.item(): [] for b in bins}
    p_entropies = []
    n_entropies = []

    binned_log_probs = {b.item(): [] for b in bins}
    p_log_probs = []
    n_log_probs = []

    seq_level_entropies = masked_mean(batch.batch["entropys"], batch.batch["response_mask"], axis=-1)
    seq_level_log_probs = masked_sum(batch.batch["old_log_probs"], batch.batch["response_mask"], axis=-1)
    seq_level_scores = masked_sum(batch.batch["token_level_scores"], batch.batch["response_mask"], axis=-1)
    uids = batch.non_tensor_batch["uid"]

    for uid in set(uids):
        # Get indices at which the current uid appears (leveraging that uids is a numpy array)
        indices = (uids == uid).nonzero()[0]
        rewards = seq_level_scores[indices]
        entropies = seq_level_entropies[indices]
        log_probs = seq_level_log_probs[indices]
        
        # Add the average entropy and log-probability in the bin corresponding to the accuracy
        avg_group_acc = rewards.mean().item()
        avg_group_entropy = entropies.mean().item()
        avg_group_log_prob = log_probs.mean().item()
        for b in bins:
            if avg_group_acc <= b:
                binned_entropies[b.item()].append(avg_group_entropy)
                binned_log_probs[b.item()].append(avg_group_log_prob)
                break

        # Add to positive/negative entropies and log_probs
        p_indices = indices[rewards > 0]
        n_indices = indices[rewards <= 0]
        p_entropies.extend(seq_level_entropies[p_indices].tolist())
        n_entropies.extend(seq_level_entropies[n_indices].tolist())
        p_log_probs.extend(seq_level_log_probs[p_indices].tolist())
        n_log_probs.extend(seq_level_log_probs[n_indices].tolist())

    p_entropies = torch.tensor(p_entropies)
    n_entropies = torch.tensor(n_entropies)
    p_log_probs = torch.tensor(p_log_probs)
    n_log_probs = torch.tensor(n_log_probs)

    metrics = {
        "p_entropies/mean": p_entropies.mean().item(),
        "p_entropies/std": p_entropies.std().item(),
        "n_entropies/mean": n_entropies.mean().item(),
        "n_entropies/std": n_entropies.std().item(),
        "p_log_probs/mean": p_log_probs.mean().item(),
        "p_log_probs/std": p_log_probs.std().item(),
        "n_log_probs/mean": n_log_probs.mean().item(),
        "n_log_probs/std": n_log_probs.std().item(),
        **{"binned_entropies/mean/" + str(k): torch.tensor(v).mean().item() for k, v in binned_entropies.items()},
        **{"binned_entropies/std/" + str(k): torch.tensor(v).std().item() for k, v in binned_entropies.items()},
        **{"binned_log_probs/mean/" + str(k): torch.tensor(v).mean().item() for k, v in binned_log_probs.items()},
        **{"binned_log_probs/std/" + str(k): torch.tensor(v).std().item() for k, v in binned_log_probs.items()},
        
    }

    return {f"{prefix}/{k}": v for k, v in metrics.items()}