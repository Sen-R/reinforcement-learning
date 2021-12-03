"""Module defines Agent."""

from typing import Any
from rl.policies.base import Policy


class Agent:
    """Agent that interacts with environments.

    `Agent` simply takes care of interaction with the environment (making sure
    actions and rewards take place in right order, storing state between
    events, etc); the brains of the operation is embedded in the agent's
    `policy`.

    Agent will also feedback the reward signal from the environment (along
    with associated state and action) to enable to policy to update itself
    (if it so desires).

    Args:
      policy: the `Policy` the agent should follow (and potentially inform
        about rewards).
    """

    def __init__(self, policy: Policy):
        self.policy = policy

    def action(self, state) -> Any:
        """Requests desired action from the agent given state signal.

        Agent in turn requests an `ActionSelector` from its embedded policy,
        which it calls to select a concrete action to return.
        """
        # Check that agent is in right state to deliver action
        # (i.e. it's not expecting a reward)
        if hasattr(self, "last_state") or hasattr(self, "last_action"):
            raise RuntimeError(
                "agent hasn't been sent a reward signal for previous action"
            )

        # Select an action to return given the observed state signal.
        action_selector = self.policy(state)
        chosen_action = action_selector()

        # Save observed state and chosen action for use when processing
        # reward signal to be obtained from environment
        self.last_state = state
        self.last_action = chosen_action

        # Done
        return chosen_action

    def reward(self, reward) -> None:
        """Sends reward signal `reward` to the agent."""
        # Check that the agent is in right state to process reward signal,
        # i.e. it has selected an action already and hasn't yet received
        # reward.
        if not (hasattr(self, "last_state") or hasattr(self, "last_action")):
            raise RuntimeError(
                "agent's hasn't yet selected an action, so not ready to "
                "process any reward"
            )

        # Update the policy
        self.policy.update(self.last_state, self.last_action, reward)

        # Clear last state and action fields to indicate agent is ready
        # to determine its next action.
        del self.last_state, self.last_action
