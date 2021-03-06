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




The option chain can also be used to find a distribution

.. code-block:: python
   

   sym = 'SPY'
   typ = 'call'

   # fuck you J Powell
   riskFree = 0.005

   # Current price
   px = 340.0

   # Use your own date
   query=['xdate-gte:20200911','put_call-eq:{}'.format(typ)]

   # I unabashedly rep my own PyAlly package
   #  Check it out at https://alienbrett.github.io/PyAlly/
   ops = ally.Ally().search(
      sym,
      query = query,
      fields = ['strikeprice','imp_Volatility','bid', 'ask', 'days_to_expiration']
   )

   # Generate the model
   m = rraider.forecast.ImpliedOptionModel(
      # Strike prices
      strikes = ops['strikeprice'],

      # The natural implied volatility
      #  This is fed to the smoother
      impVol = ops['imp_Volatility'],

      bid = ops['bid'],
      ask = ops['ask'],

      # Are these calls or puts? this impacts
      # option pricing later
      callput = typ,

      # risk free rate
      irate = riskFree,

      # Time, but it's wise to anualize it here
      time = ops['days_to_expiration']/ 252.0,

      # Current price
      spot = px,

      # Use Gaussian 4.0-wide kernel to smooth strikes [Default]
      # None uses default
      volSmoother = None,
   )

   import scipy.ndimage

   # We can also customize the volatility smoother here
   # Any kernel or smoother function will do
   m.setVolSmoother(
       lambda vol: scipy.ndimage.gaussian_filter1d ( vol, sigma=4.0 )
   )


   # Now we can display the distribution
   #  This is currently a little round-a-bout, but
   #  Hopefully will be cleaner in the future

   # Which time to consider
   n = 0
   plt.hist(
      m.simulate(m.times[n], n=1000,),
      bins=25,
      density=True
   )
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



Implied Option model

.. autoclass:: rraider.forecast.ImpliedOptionModel
   :members:
