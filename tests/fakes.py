from typing import Final
from rl.policies.base import DumbPolicy
from rl import Agent
from rl.action_selectors import DeterministicActionSelector
from rl.environments.base import Environment


class FakePolicy(DumbPolicy):
    def __init__(self, action_to_always_return: int):
        self.action_selector: Final = DeterministicActionSelector(
            action_to_always_return
        )
        self._state = 0  # "state" is call count

    def __call__(self, state) -> DeterministicActionSelector:
        self._state += 1
        return self.action_selector

    @property
    def state(self) -> dict:
        return {"call_count": self._state}


def fake_agent(action_to_always_return: int = 0) -> Agent:
    policy = FakePolicy(action_to_always_return)
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
        if self.episode_length is None:
            return False
        else:
            return self.t >= self.episode_length
