from abc import ABC
import scipy
import numpy as np
import typing
from ioh import get_problem, logger, ProblemClass


class VerticalModelEvaluator(ABC):

    def __init__(self, problem: ProblemClass, minimal_anchor: int, final_anchor: int) -> None:
        """
        Initialises the vertical model evaluator. Take note of what the arguments are
        
        :param surrogate_model: A sklearn pipeline object, which has already been fitted on LCDB data. You can use the predict model to predict for a numpy array (consisting of configuration information and an anchor size) what the performance of that configuration is. 
        :param minimal_anchor: Smallest anchor to be used
        :param final_anchor: Largest anchor to be used
        """
        self.problem = problem
        self.minimal_anchor = minimal_anchor
        self.final_anchor = final_anchor

    def evaluate_model(self, best_so_far: None|float, configuration: typing.Dict) -> typing.List[typing.Tuple[int, float]]:
        raise NotImplementedError()


class IPL(VerticalModelEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def inverse_power_law(x, a, b):
        f_x = np.log(x)/np.log(a) + b
        return f_x

    @staticmethod
    def fit_inverse_power(performance: typing.List[typing.Tuple[int, float]]):
        # Fit inverse power law function to observed data
        x = []
        y = []
        for xi, perf in performance:
            x.append(xi)
            y.append(perf)

        popt, pcov = scipy.optimize.curve_fit(IPL.inverse_power_law, x, y)
        return popt, pcov

    def evaluate_model(self, best_so_far: typing.Optional[float], configuration: typing.Dict) -> typing.List[typing.Tuple[int, float]]:
        # This function needs to be implemented

        if best_so_far is None: #If no best yet, evaluate with external surrogate at max anchor size
            return [(self.final_anchor, self.problem(configuration))]
        
        train_anchors = np.linspace(0, self.final_anchor*0.4, 10) # We will evaluate 10 anchor sizes uptill 0.4*max_anchor 
        results = []
        for anchor in train_anchors:
            results.append((anchor, self.problem(configuration))) 
        
        popt, pcov = IPL.fit_inverse_power(performance=results) # Fit inverse power law (IPL) function to data
        best = IPL.inverse_power_law(self.final_anchor, popt[0], popt[1]) # Predict performance at max anchor size using IPL
        if best < best_so_far:
            results.append((self.final_anchor, best))
        return results