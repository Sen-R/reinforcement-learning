import unittest
from rlbook.environments.base import Environment
from rlbook.agents.base import Agent
from rlbook.simulator import SingleAgentWaitingSimulator, History


mock_tape = []


class MockAgent(Agent):
    """Mock agent that logs calls to its methods"""

    def __init__(self, tape):
        self.tape = tape
        self.action_to_return = 0

    def action(self, state):
        self.tape.append(("action", state, self.action_to_return))
        return self.action_to_return

    def reward(self, r):
        self.tape.append(("reward", r))
        pass


class MockEnvironment(Environment):
    """Mock environment that logs calls to its methods."""

    def __init__(self, tape):
        self.tape = tape
        self.state_to_return = 1.0
        self.reward_to_return = 10.0

    def state(self):
        self.tape.append(("state", self.state_to_return))
        return self.state_to_return

    def act(self, a):
        self.tape.append(("act", a, self.reward_to_return))
        return self.reward_to_return


def create_environment():
    agent = MockAgent(mock_tape)
    environment = MockEnvironment(mock_tape)
    return SingleAgentWaitingSimulator(environment, agent)


class TestSingleAgentWaitingSimulator(unittest.TestCase):
    def test_init(self):
        sim = create_environment()
        self.assertIsInstance(sim.environment, Environment)
        self.assertIsInstance(sim.agent, Agent)
        self.assertEqual(len(sim.history.states), 0)
        self.assertEqual(len(sim.history.actions), 0)
        self.assertEqual(len(sim.history.rewards), 0)
        self.assertEqual(sim.t, 0)

    def test_run_history(self):
        n_steps = 5
        sim = create_environment()
        sim.run(n_steps)
        self.assertEqual(len(sim.history.states), n_steps)
        self.assertEqual(len(sim.history.actions), n_steps)
        self.assertEqual(len(sim.history.rewards), n_steps)
        self.assertEqual(sim.t, n_steps)

    def test_run_env_agent_interactions_are_correct(self):
        n_steps = 1
        mock_tape.clear()
        sim = create_environment()
        agent, environment = sim.agent, sim.environment
        sim.run(n_steps)
        expected_tape = [
            ("state", environment.state_to_return),
            ("action", environment.state_to_return, agent.action_to_return),
            ("act", agent.action_to_return, environment.reward_to_return),
            ("reward", environment.reward_to_return),
        ]
        self.assertEqual(mock_tape, expected_tape)
        self.assertEqual(sim.history.states[0], environment.state_to_return)
        self.assertEqual(sim.history.actions[0], agent.action_to_return)
        self.assertEqual(sim.history.rewards[0], environment.reward_to_return)


class TestHistory(unittest.TestCase):
    def test_init(self):
        h = History()
        self.assertEqual(len(h.states), 0)
        self.assertEqual(len(h.actions), 0)
        self.assertEqual(len(h.rewards), 0)

    def test_add(self):
        h = History()
        s, a, r = [1.0], 3, 0.5
        h.add(s, a, r)
        self.assertEqual(h.states, [s])
        self.assertEqual(h.actions, [a])
        self.assertEqual(h.rewards, [r])
