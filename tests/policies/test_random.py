import pytest
import numpy as np
from rl.policies.random import DiscreteRandomPolicy
from rl.action_selectors import UniformDiscreteActionSelector


class TestDiscreteRandomPolicy:
    def test_policy_returns_uniform_discrete_action_selector(self) -> None:
        k = 5  # arbitrary
        policy = DiscreteRandomPolicy(k)
        action_selector = policy(state=None)
        assert isinstance(action_selector, UniformDiscreteActionSelector)
        assert action_selector.n_actions == k

    @pytest.mark.parametrize(
        "state", [None, 3, [1, 2], np.array([[1.0, 2.0], [3.0, 4.0]])]
    )
    def test_policy_works_with_different_state_signals(self, state) -> None:
        """Tests whether policy's call method works with variety of state
        signals."""
        policy = DiscreteRandomPolicy(5)
        action = policy(state)
        assert isinstance(action, UniformDiscreteActionSelector)

    def test_state_attr_returns_empty_dict(self) -> None:
        """State method should return empty dict as this policy has no
        parameters."""
        policy = DiscreteRandomPolicy(5)
        assert policy.state == {}
