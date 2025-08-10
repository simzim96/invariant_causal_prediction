Usage
=====

Install
-------

.. code-block:: bash

   pip install invariant-causal-prediction

Quick start
-----------

.. code-block:: python

   import numpy as np
   from invariant_causal_prediction import icp

   rng = np.random.default_rng(0)
   X = rng.standard_normal((200, 3))
   Exp = np.r_[np.zeros(100), np.ones(100)]
   X[Exp==1] *= 1.2
   y = X[:,0] + rng.standard_normal(200)
   res = icp(X, y, Exp)
   print(res['conf_int']) 