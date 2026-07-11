PreML Documentation
====================

PreML provides statistically grounded EDA, preprocessing, and recommendation helpers for tabular data.

Getting Started
---------------

.. code-block:: python

   import pandas as pd
   from preml.recommendation_engine import RecommendationEngine

   X = pd.DataFrame({"age": [21, 35, 42], "city": ["A", "B", "A"]})
   y = pd.Series([0, 1, 0])

   engine = RecommendationEngine(random_state=42)
   result = engine.get_recommendation(X, y)
   print(result["model"])

API Reference
-------------

.. toctree::
   :maxdepth: 2

   recommendation_engine

Interactive Examples
--------------------

This project can be paired with nbsphinx or MyST-NB for notebook-based examples in a full documentation build.
