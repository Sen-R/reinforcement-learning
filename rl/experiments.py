"""Objects and functions to run and keep track of experiments."""

from typing import Callable, Dict, Any
import functools


class ExperimentalRun:
    """Class representing a single parameterised run of an experiment.

    Wraps an inner experimental `methodology` (a callable) in order to:
    1. Store the parameters used for this concrete experimental run
    2. Defer running the experiment until this is required (`run` method)
    3. Store the results of running the experiment

    Args:
      methodology: the experimental methodology that will be run
      parameters: the parameters to use for this concrete instantiation of
        that methodology
    """

    def __init__(self, methodology: Callable, parameters: Dict):
        self._parameters = parameters
        self.methodology = methodology

    @property
    def parameters(self) -> Dict:
        """Return the parameters for this particular experimental run."""
        return self._parameters

    def run(self) -> None:
        """Execute this run of the experiment and store results."""
        self._results = self.methodology(**self.parameters)

    @property
    def results(self) -> Any:
        """Retrieve the results."""
        if not hasattr(self, "_results"):
            raise RuntimeError(
                "No results available as experiment has not yet been run"
            )
        return self._results


def methodology(methodology_function: Callable):
    """Decorator converting methodologies to experiment builders.

    This decorate takes a callable that represents an experimental
    methodology -- formally a mapping from a dictionary of experimental
    parameters to an object representing experimental results -- and converts
    it into an "experiment builder".

    By "experiment builder" we mean another callable that can be passed
    the parameters corresponding to a single run which will return a
    corresponding ExperimentalRun object for you.

    Example:
      # Define a "methodology" consisting of taking a single parameter
      # and squaring it. The following returns an experiment builder
      # wrapping this methodology
      @methodology
      def squaring_experiment(x): x ** 2

      # Create multiple experimental runs (with different parameters)
      # using this methodology
      experimental_runs = [squaring_experiment(x=x) for x in range(5)]

      # Run these in turn and print the results
      for e in experimental_runs:
          e.run()
          print(e.results)

    Args:
      methodology_function: The methodology to wrap.

    Returns:
      A new callable that represents an experimental run builder as described.
    """

    @functools.wraps(methodology_function)
    def experimental_run_builder(**parameters):
        return ExperimentalRun(methodology_function, parameters)

    return experimental_run_builder
