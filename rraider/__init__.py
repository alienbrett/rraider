"""

Goal
	Set of well-documented and easy to use tool functions that
		provide guidance when placing trades manually.

Needed

	1)	Data Acquisition
		( ) Mostly turn-key
		( ) Caching responses

	2)	Simulators
		(x) Brownian motion
		(x) Geometric brownian motion

	3)	Stat Parameter estimation
		(x) Volatility
		( ) Correlation metrics

	4)	Pricing/Execution/Risk Tools
		(x) % prob of executing specific trade
			simulation.fullExecEstimator(sims,spread)
		( ) Analyze exposure to statistical risk factors
        (x) Create bracket positions with capped risk

	5)	Well-defined strategic trade functions
		(x) Minimum-variance portfolio allocations


"""
from . import data
from . import estimate
from . import portfolio
from . import pricing
from . import simulate
from . import utils
from . import risk
from . import BlackScholes as bs
