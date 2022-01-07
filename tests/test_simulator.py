from typing import List, Tuple, Final
from .fakes import FakePolicy, FakeEnvironment, fake_agent
from rl.environments.base import Environment
from rl import Agent
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

    def observe(self) -> float:
        self.tape.append(("observe", self.state_to_return))
        return self.state_to_return

    def act(self, a) -> float:
        self.tape.append(("act", a, self.reward_to_return))
        return self.reward_to_return

    def reset(self, random_state=None) -> None:
        self.tape.append(("reset",))

    @property
    def done(self) -> bool:
        self.tape.append(("done",))
        return False


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
            ("observe", s),
            ("action", s, a),
            ("act", a, r),
            ("reward", r),
            ("done",),
        ]
        assert mock_tape == expected_tape
        assert sim.history.states[0] == s
        assert sim.history.actions[0] == a
        assert sim.history.rewards[0] == r

    def test_breaks_when_done(self) -> None:
        """Tests whether simulation loop breaks when environment says
        terminal state has been reached."""
        # Create an environment with episode length 2 and a fake agent
        # to interact with it.
        episode_length = 2
        environment = FakeEnvironment(episode_length=episode_length)
        agent = fake_agent()
        sim = SingleAgentWaitingSimulator(environment, agent)

        # Run for many more timesteps than the episode length and check
        # that simulation actually stopped after 2 steps
        run_length = 102  # much greater than epsiode length
        sim.run(run_length)
        assert sim.t == episode_length


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

    def test_to_dict(self) -> None:
        h = History()
        s, a, r = [1.0], 3, 0.5
        h.add(s, a, r)
        expected_dict = {"states": [s], "actions": [a], "rewards": [r]}
        actual_dict = h.to_dict()
        assert actual_dict == expected_dict

        # Also check that data has been copied to create the dict
        assert h.states is not actual_dict["states"]
        assert h.actions is not actual_dict["actions"]
        assert h.rewards is not actual_dict["rewards"]
