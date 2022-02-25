from typing import TypeVar, Collection, Tuple, Mapping, Callable


State = TypeVar("State")
Action = TypeVar("Action")
Reward = float
Probability = float
Policy = Callable[[State], Collection[Tuple[Action, Probability]]]
NextStateRewardAndProbability = Tuple[State, Reward, Probability]
TransitionsMapping = Mapping[
    State, Mapping[Action, Collection[NextStateRewardAndProbability[State]]]
]
