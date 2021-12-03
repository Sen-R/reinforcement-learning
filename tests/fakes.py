from typing import Final
from rl.policies.base import DumbPolicy
from rl.agent import Agent
from rl.action_selector import DeterministicActionSelector


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
