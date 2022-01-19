from numpy.testing import assert_array_equal
from rl.callbacks import History, AgentStateLogger, EnvironmentStateLogger
from rl.simulator import SingleAgentWaitingSimulator
from .fakes import FakeEnvironment, fake_agent


class TestHistory:
    def test_init(self) -> None:
        h = History()
        assert len(h.states) == 0
        assert len(h.actions) == 0
        assert len(h.rewards) == 0
        assert h.logging_period == 1

    def test_call(self) -> None:
        h = History()
        environment = FakeEnvironment(reward_to_return=1.0)
        agent = fake_agent()
        sim = SingleAgentWaitingSimulator(environment, agent)
        s, a, r, d = [1.0], 3, 0.5, False
        h(sim, s, a, r, d)
        assert h.states == [s]
        assert h.actions == [a]
        assert h.rewards == [r]

    def test_to_dict(self) -> None:
        h = History()
        environment = FakeEnvironment(reward_to_return=1.0)
        agent = fake_agent()
        sim = SingleAgentWaitingSimulator(environment, agent)
        s, a, r, d = [1.0], 3, 0.5, False
        h(sim, s, a, r, d)
        expected_dict = {"states": [s], "actions": [a], "rewards": [r]}
        actual_dict = h.to_dict()
        assert actual_dict == expected_dict

        # Also check that data has been copied to create the dict
        assert h.states is not actual_dict["states"]
        assert h.actions is not actual_dict["actions"]
        assert h.rewards is not actual_dict["rewards"]

    def test_functional(self) -> None:
        STATE = (1, 2, 3)
        ACTION = 5
        REWARD = 1.0
        EPISODE_LENGTH = 10
        LOGGING_PERIOD = 5
        environment = FakeEnvironment(
            state_to_return=STATE,
            reward_to_return=REWARD,
            episode_length=EPISODE_LENGTH,
        )
        agent = fake_agent(action_to_always_return=ACTION)
        history = History(logging_period=5)
        sim = SingleAgentWaitingSimulator(
            environment, agent, callbacks=[history]
        )
        sim.run(EPISODE_LENGTH + 100)
        expected_length = EPISODE_LENGTH // LOGGING_PERIOD
        assert_array_equal(history.states, [STATE] * expected_length)
        assert_array_equal(history.actions, [ACTION] * expected_length)
        assert_array_equal(history.rewards, [REWARD] * expected_length)


class TestAgentStateLogger:
    def test_functional(self):
        environment = FakeEnvironment(reward_to_return=1.0)
        agent = fake_agent()
        callbacks = [AgentStateLogger(logging_period=5)]
        sim = SingleAgentWaitingSimulator(
            environment, agent, callbacks=callbacks
        )
        sim.run(10)
        assert callbacks[0].states == [{"call_count": s} for s in [5, 10]]

    def test_default_period_is_one(self):
        callback = AgentStateLogger()
        assert callback.logging_period == 1


class TestEnvironmentStateLogger:
    def test_functional(self):
        environment = FakeEnvironment()
        agent = fake_agent()
        callbacks = [EnvironmentStateLogger(logging_period=5)]
        sim = SingleAgentWaitingSimulator(
            environment, agent, callbacks=callbacks
        )
        sim.run(15)
        assert callbacks[0].states == [
            {"action_count": s} for s in [5, 10, 15]
        ]

    def test_default_period_is_one(self):
        callback = EnvironmentStateLogger()
        assert callback.logging_period == 1
