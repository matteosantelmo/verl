# Copyright 2025 Individual Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Diversity bonus computation for reinforcement learning training.

This module provides functionality to add a diversity bonus to rewards,
encouraging diversity among model outputs for the same prompt.
"""

from collections import defaultdict
from typing import Callable, Optional

import torch


# Type alias for the diversity metric function
# Takes two response strings and returns a diversity score between 0 and 1
DiversityMetricFn = Callable[[str, str], float]

# Registry for diversity metrics
_DIVERSITY_METRIC_REGISTRY: dict[str, DiversityMetricFn] = {}


def register_diversity_metric(name: str):
    """Decorator to register a diversity metric function.
    
    Usage:
        @register_diversity_metric("my_metric")
        def my_diversity_metric(response1: str, response2: str) -> float:
            ...
    """
    def decorator(fn: DiversityMetricFn) -> DiversityMetricFn:
        _DIVERSITY_METRIC_REGISTRY[name] = fn
        return fn
    return decorator


def get_diversity_metric(name: str) -> DiversityMetricFn:
    """Get a diversity metric by name from the registry.
    
    Args:
        name: The registered name of the diversity metric.
    
    Returns:
        The diversity metric function.
    
    Raises:
        KeyError: If the metric name is not found in the registry.
    """
    if name not in _DIVERSITY_METRIC_REGISTRY:
        available = list(_DIVERSITY_METRIC_REGISTRY.keys())
        raise KeyError(
            f"Diversity metric '{name}' not found. Available metrics: {available}"
        )
    return _DIVERSITY_METRIC_REGISTRY[name]


def compute_pairwise_diversity(
    response: str,
    other_responses: list[str],
    diversity_metric: DiversityMetricFn,
) -> float:
    """
    Compute the average pairwise diversity of a response against other responses.
    
    Args:
        response: The response string to evaluate.
        other_responses: List of other response strings to compare against.
        diversity_metric: Function that takes two strings and returns diversity score [0, 1].
    
    Returns:
        Average diversity score. Returns 0.0 if there are no other responses.
    """
    if not other_responses:
        return 0.0
    
    total_diversity = sum(diversity_metric(response, other) for other in other_responses)
    return total_diversity / len(other_responses)


def compute_diversity_bonus(
    responses: list[str],
    rewards: list[float],
    uids: list[str],
    diversity_metric: DiversityMetricFn,
    lambda_pos: float = 0.0,
    lambda_neg: float = 0.0,
    reward_threshold: float = 0.5,
) -> list[float]:
    """
    Compute diversity bonus for each response based on its diversity within its group.
    
    Responses are grouped by uid (same prompt) and split into correct/incorrect
    based on whether their reward exceeds the threshold. Diversity is measured
    only within each correctness group (correct vs correct, incorrect vs incorrect).
    
    Args:
        responses: List of response strings.
        rewards: List of reward values corresponding to each response.
        uids: List of unique identifiers grouping responses by prompt.
        diversity_metric: Function that takes two strings and returns diversity score [0, 1].
        lambda_pos: Scaling factor for diversity bonus on correct responses.
        lambda_neg: Scaling factor for diversity bonus on incorrect responses.
        reward_threshold: Threshold to determine correct (>= threshold) vs incorrect responses.
    
    Returns:
        List of diversity bonus values for each response.
    """
    n = len(responses)
    if n == 0:
        return []
    
    # Early exit if diversity bonus is disabled
    if lambda_pos == 0.0 and lambda_neg == 0.0:
        return [0.0] * n
    
    # Group responses by uid
    uid_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, uid in enumerate(uids):
        uid_to_indices[uid].append(i)
    
    # Compute diversity bonus for each response
    bonuses = [0.0] * n
    
    for uid, indices in uid_to_indices.items():
        if len(indices) <= 1:
            # No diversity bonus for single responses
            continue
        
        # Split into correct and incorrect groups
        correct_indices = [i for i in indices if rewards[i] > reward_threshold]
        incorrect_indices = [i for i in indices if rewards[i] <= reward_threshold]
        
        # Compute diversity bonus for correct responses
        if len(correct_indices) > 1 and lambda_pos != 0.0:
            correct_responses = [responses[i] for i in correct_indices]
            for idx, i in enumerate(correct_indices):
                other_responses = correct_responses[:idx] + correct_responses[idx + 1:]
                diversity = compute_pairwise_diversity(
                    responses[i], other_responses, diversity_metric
                )
                bonuses[i] = lambda_pos * diversity
        
        # Compute diversity bonus for incorrect responses
        if len(incorrect_indices) > 1 and lambda_neg != 0.0:
            incorrect_responses = [responses[i] for i in incorrect_indices]
            for idx, i in enumerate(incorrect_indices):
                other_responses = incorrect_responses[:idx] + incorrect_responses[idx + 1:]
                diversity = compute_pairwise_diversity(
                    responses[i], other_responses, diversity_metric
                )
                bonuses[i] = lambda_neg * diversity
    
    return bonuses


class DiversityBonusComputer:
    """
    A class to compute diversity bonus for batches of responses.
    
    This class encapsulates the configuration for diversity bonus computation
    and provides a convenient interface for use in reward managers.
    """
    
    def __init__(
        self,
        diversity_metric: Optional[DiversityMetricFn] = None,
        lambda_pos: float = 0.0,
        lambda_neg: float = 0.0,
        reward_threshold: float = 0.5,
    ):
        """
        Initialize the diversity bonus computer.
        
        Args:
            diversity_metric: Function that takes two strings and returns diversity score [0, 1].
                            If None, diversity bonus is disabled.
            lambda_pos: Scaling factor for diversity bonus on correct responses.
            lambda_neg: Scaling factor for diversity bonus on incorrect responses.
            reward_threshold: Threshold to determine correct vs incorrect responses.
        """
        self.diversity_metric = diversity_metric
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        self.reward_threshold = reward_threshold
    
    @property
    def enabled(self) -> bool:
        """Check if diversity bonus is enabled."""
        return (
            self.diversity_metric is not None
            and (self.lambda_pos != 0.0 or self.lambda_neg != 0.0)
        )
    
    def compute(
        self,
        responses: list[str],
        rewards: list[float],
        uids: list[str],
    ) -> list[float]:
        """
        Compute diversity bonus for a batch of responses.
        
        Args:
            responses: List of response strings.
            rewards: List of reward values corresponding to each response.
            uids: List of unique identifiers grouping responses by prompt.
        
        Returns:
            List of diversity bonus values for each response.
        """
        if not self.enabled:
            return [0.0] * len(responses)
        
        return compute_diversity_bonus(
            responses=responses,
            rewards=rewards,
            uids=uids,
            diversity_metric=self.diversity_metric,
            lambda_pos=self.lambda_pos,
            lambda_neg=self.lambda_neg,
            reward_threshold=self.reward_threshold,
        )
    
    def apply_to_reward_tensor(
        self,
        reward_tensor: torch.Tensor,
        responses: list[str],
        rewards: list[float],
        uids: list[str],
        valid_response_lengths: list[int],
    ) -> tuple[torch.Tensor, list[float]]:
        """
        Apply diversity bonus directly to a reward tensor.
        
        Args:
            reward_tensor: The reward tensor to modify (shape: [batch_size, seq_len]).
            responses: List of response strings.
            rewards: List of base reward values corresponding to each response.
            uids: List of unique identifiers grouping responses by prompt.
            valid_response_lengths: List of valid response lengths for each sample.
        
        Returns:
            Tuple of (modified reward tensor, list of diversity bonuses).
        """
        if not self.enabled:
            return reward_tensor, [0.0] * len(responses)
        
        bonuses = self.compute(responses, rewards, uids)
        
        for i, bonus in enumerate(bonuses):
            if bonus != 0.0:
                reward_tensor[i, valid_response_lengths[i] - 1] += bonus
        
        return reward_tensor, bonuses


# Example diversity metrics (placeholder implementations)
@register_diversity_metric("dummy")
def dummy_diversity_metric(response1: str, response2: str) -> float:
    """
    Placeholder diversity metric that always returns 0.5.
    Replace with actual implementation.
    """
    return 0.5


@register_diversity_metric("char_difference")
def char_difference_diversity(response1: str, response2: str) -> float:
    """
    Simple character-level diversity metric based on normalized edit distance.
    This is a basic example - replace with more sophisticated metrics.
    """
    if not response1 and not response2:
        return 0.0
    if not response1 or not response2:
        return 1.0
    
    # Simple normalized character difference
    max_len = max(len(response1), len(response2))
    if max_len == 0:
        return 0.0
    
    # Count different characters at each position (simple approximation)
    min_len = min(len(response1), len(response2))
    diff_count = sum(1 for i in range(min_len) if response1[i] != response2[i])
    diff_count += abs(len(response1) - len(response2))
    
    return min(1.0, diff_count / max_len)


@register_diversity_metric("math_eq_em")
def math_equation_match_diversity(response1: str, response2: str) -> float:
	"""
	Simple diversity metric for mathematical responses based on final answer extraction.
	"""
	from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed, strip_string

	def get_canonical_answer(solution_str):
		"""
		Extracts the final answer from a solution string and normalizes it
		using the provided strip_string method.
		
		Returns:
			str: The normalized answer string, or None if extraction fails.
		"""
		try:
			# Extract the content inside the last \boxed{}
			boxed_content = last_boxed_only_string(solution_str)
			# Remove the \boxed command itself
			answer = remove_boxed(boxed_content)
			# Normalize the answer string
			answer = strip_string(answer)
			return answer
		except Exception as e:
			return answer

	canonical1 = get_canonical_answer(response1)
	canonical2 = get_canonical_answer(response2)
	return 0.0 if canonical1 == canonical2 else 1.0