Forecast
==========


Quick examples
----------------


Multi-asset models can be used to model several securities and the
correlation between them. For example, log-normal distributions distributed in the
usual fashion can be modeled as such:



.. code-block:: python

   # Current price of assets
   # Assume they have same price, for side-by-side comparison
   price = 100

   # Anualized means of two different assets
   means = [0.01, 0.06]

   # Anualized covariance/dispersion matrix
   vol = [[0.15, -0.08], [-0.08, 0.82]]

   # How many discrete time points to simulate for each T
   # For a single year, it might make sense to use 252, one unit for each trading day
   # Or for shorter intervals, sample several times a day (10, for example)
   resolution = 252 * 20

   m = rraider.forecast.MultiLogBrownianModel(
       S0 = price,
       mu = means,
       cov = vol,
       N = resolution
   )
   # Look 10 trading days out, about 2 weeks
   x,y = m.simulate(horizon=10/252.0, n=1, path=True)

   # Plot asset 1
   for i in (0,1):
	   plt.plot(x, y[0,:,i])
   plt.show()


Base class
----------

Both derive from ``AssetModel``, and all have the same
``simulate`` and ``distribution`` functions.

.. autoclass:: rraider.forecast.AssetModel


Selecting a model
-------------------

There are currently two types of asset models,

- Single-asset models
- Multi-asset models



Brownian models

.. autoclass:: rraider.forecast.BrownianModel
   :members: __init__
.. autoclass:: rraider.forecast.MultiBrownianModel
   :members: __init__


Log-normal Brownian models

.. autoclass:: rraider.forecast.LogBrownianModel
   :members: __init__
.. autoclass:: rraider.forecast.MultiLogBrownianModel
   :members: __init__


Garch volatility model

.. autoclass:: rraider.forecast.GarchModel
   :members: __init__
