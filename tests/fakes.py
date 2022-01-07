from typing import Final
from rl.policies.base import DumbPolicy
from rl.agent import Agent
from rl.action_selectors import DeterministicActionSelector
from rl.environments.base import Environment


class FakePolicy(DumbPolicy):
    def __init__(self, action_to_always_return):
        self.action_selector: Final = DeterministicActionSelector(
            action_to_always_return
        )

    def __call__(self, state) -> DeterministicActionSelector:
        return self.action_selector


def fake_agent():
    policy = FakePolicy(0)
    return Agent(policy)


class FakeEnvironment(Environment):
    """Fake environment that always returns same state and reward irrespective
    of action taken and lasts for `episode_length` time steps."""

    def __init__(
        self, state_to_return=0, reward_to_return=0.0, episode_length=None
    ):
        self.state_to_return = state_to_return
        self.reward_to_return = reward_to_return
        self.episode_length = episode_length
        self.reset()

    def act(self, action):
        self.t += 1
        return self.reward_to_return

    def observe(self):
        return self.state_to_return

    def reset(self, random_state=None) -> None:
        self.t = 0

    @property
    def done(self) -> bool:
        return self.t >= self.episode_length
