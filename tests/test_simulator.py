from typing import List, Tuple, Final
from .fakes import FakePolicy, FakeEnvironment, fake_agent
from rl.environments.base import Environment
from rl import Agent
from rl.simulator import SingleAgentWaitingSimulator
from rl.callbacks import Callback


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

    @property
    def state(self) -> float:
        self.tape.append(("state",))
        return self.state_to_return


class MockCallback(Callback):
    """Mock callback to supply to simulator."""

    def __init__(self, name: str, tape: List[Tuple]):
        self.name = name
        self.tape = tape

    def __call__(
        self,
        sim: SingleAgentWaitingSimulator,
        state,
        action,
        reward,
        done: bool,
    ) -> None:
        self.tape.append((self.name, sim, state, action, reward, done))


def create_environment() -> SingleAgentWaitingSimulator:
    agent = MockAgent(mock_tape)
    environment = MockEnvironment(mock_tape)
    callbacks = [MockCallback(name, mock_tape) for name in ["cb0", "cb1"]]
    return SingleAgentWaitingSimulator(environment, agent, callbacks=callbacks)


class TestSingleAgentWaitingSimulator:
    def test_init(self) -> None:
        sim = create_environment()
        assert isinstance(sim.environment, Environment)
        assert isinstance(sim.agent, Agent)
        assert sim.t == 0

    def test_run_history(self) -> None:
        n_steps = 5
        sim = create_environment()
        sim.run(n_steps)
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
            ("cb0", sim, s, a, r, False),
            ("cb1", sim, s, a, r, False),
        ]
        assert mock_tape == expected_tape

    def test_default_simulator_has_no_callbacks(self) -> None:
        agent = fake_agent()
        environment = FakeEnvironment()
        sim = SingleAgentWaitingSimulator(environment, agent)
        assert len(sim.callbacks) == 0

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
