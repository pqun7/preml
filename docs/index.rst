PreML Documentation
====================

PreML provides statistically grounded EDA, preprocessing, and recommendation helpers for tabular data.

Recommended quick start:

.. code-block:: python

   import pandas as pd
   from preml import PreML

   df = pd.DataFrame({"age": [21, 35, 42], "city": ["A", "B", "A"], "target": [0, 1, 0]})
   ml = PreML(df, target="target")
   analysis = ml.analyze()
   print(ml.summary())

Getting Started
---------------

.. code-block:: python

   import pandas as pd
   from preml import PreML

   df = pd.DataFrame({"age": [21, 35, 42], "city": ["A", "B", "A"], "target": [0, 1, 0]})
   ml = PreML(df, target="target")
   result = ml.models()
   print(result["model"])

API Reference
-------------

.. toctree::
   :maxdepth: 2

   recommendation_engine

Interactive Examples
--------------------

This project can be paired with nbsphinx or MyST-NB for notebook-based examples in a full documentation build.
