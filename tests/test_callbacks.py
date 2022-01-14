from rl.callbacks import AgentStateLogger
from rl.simulator import SingleAgentWaitingSimulator
from .fakes import FakeEnvironment, fake_agent


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
