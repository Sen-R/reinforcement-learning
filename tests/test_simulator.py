from typing import List, Tuple, Final
from .fakes import FakePolicy
from rl.environments.base import Environment
from rl.agent import Agent
from rl.simulator import SingleAgentWaitingSimulator, History


mock_tape: List[Tuple] = []


class MockAgent(Agent):
    """Mock agent that logs calls to its methods"""

    action_to_return: Final = 0

    def __init__(self, tape: List[Tuple]):
        self.tape = tape
        super().__init__(FakePolicy(self.action_to_return))

    def action(self, state) -> int:
        self.tape.append(("action", state, self.action_to_return))
        return super().action(state)

    def reward(self, reward) -> None:
        self.tape.append(("reward", reward))
        super().reward(reward)


class MockEnvironment(Environment):
    """Mock environment that logs calls to its methods."""

    state_to_return: Final = 1.0
    reward_to_return: Final = 10.0

    def __init__(self, tape: List[Tuple]):
        self.tape = tape

    def state(self) -> float:
        self.tape.append(("state", self.state_to_return))
        return self.state_to_return

    def act(self, a) -> float:
        self.tape.append(("act", a, self.reward_to_return))
        return self.reward_to_return

    def reset(self, random_state=None) -> None:
        self.tape.append(("reset",))


def create_environment() -> SingleAgentWaitingSimulator:
    agent = MockAgent(mock_tape)
    environment = MockEnvironment(mock_tape)
    return SingleAgentWaitingSimulator(environment, agent)


class TestSingleAgentWaitingSimulator:
    def test_init(self) -> None:
        sim = create_environment()
        assert isinstance(sim.environment, Environment)
        assert isinstance(sim.agent, Agent)
        assert len(sim.history.states) == 0
        assert len(sim.history.actions) == 0
        assert len(sim.history.rewards) == 0
        assert sim.t == 0

    def test_run_history(self) -> None:
        n_steps = 5
        sim = create_environment()
        sim.run(n_steps)
        assert len(sim.history.states) == n_steps
        assert len(sim.history.actions) == n_steps
        assert len(sim.history.rewards) == n_steps
        assert sim.t == n_steps

    def test_run_env_agent_interactions_are_correct(self) -> None:
        n_steps = 1
        mock_tape.clear()
        sim = create_environment()
        s = MockEnvironment.state_to_return
        a = MockAgent.action_to_return
        r = MockEnvironment.reward_to_return
        sim.run(n_steps)
        expected_tape = [
            ("state", s),
            ("action", s, a),
            ("act", a, r),
            ("reward", r),
        ]
        assert mock_tape == expected_tape
        assert sim.history.states[0] == s
        assert sim.history.actions[0] == a
        assert sim.history.rewards[0] == r


class TestHistory:
    def test_init(self) -> None:
        h = History()
        assert len(h.states) == 0
        assert len(h.actions) == 0
        assert len(h.rewards) == 0

    def test_add(self) -> None:
        h = History()
        s, a, r = [1.0], 3, 0.5
        h.add(s, a, r)
        assert h.states == [s]
        assert h.actions == [a]
        assert h.rewards == [r]
