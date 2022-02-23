import pytest
from rl.experiments import methodology, ExperimentalRun


def test_functional() -> None:
    # Create a pretend experimental methodology
    @methodology
    def my_methodology(a, b):
        return {"c": a + b}

    # Create a concrete experiment by specifying some parameters
    experiment = my_methodology(a=0, b=2)

    # Test that parameters are correctly recorded
    assert experiment.parameters == {"a": 0, "b": 2}

    # Run the experiment
    experiment.run()

    # Test that results are correctly recorded
    assert experiment.results == {"c": 2}


class TestExperimentalRun:
    def test_results_will_raise_if_experiment_not_yet_run(self) -> None:
        # Create a dummy experimental run
        experiment = ExperimentalRun(
            methodology=lambda x: x**2, parameters={"x": 2}
        )

        # If we try to access results before running it, this is a RuntimeError
        with pytest.raises(RuntimeError):
            experiment.results

        # But after running it, we are able to access results without error
        experiment.run()
        assert experiment.results == 4


class TestMethodology:
    def test_wrapped_methodology_has_inner_functions_docstring(self) -> None:
        # Define a methodology containing a docstring
        @methodology
        def my_methodology(x):
            """Methodology docstring"""
            return x

        # Check that the wrapped methodology's docstring is as expected
        assert my_methodology.__doc__ == "Methodology docstring"
