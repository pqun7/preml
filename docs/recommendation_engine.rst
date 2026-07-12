Recommendation Engine
=====================

.. automodule:: preml.recommendation_engine
   :members:
   :undoc-members:
   :show-inheritance:

Example
-------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from preml import PreML
   from preml.recommendation_engine import RecommendationEngine

   X = pd.DataFrame({
       "feature1": np.random.randn(100),
       "feature2": np.random.randn(100),
       "category": np.random.choice(["A", "B"], 100),
   })
   y = (X["feature1"] + X["feature2"] > 0).astype(int)

   df = X.copy()
   df["target"] = y
   ml = PreML(df, target="target")
   print(ml.analyze()["data_quality_score"])

   engine = RecommendationEngine(random_state=42)
   recommendation = engine.generate_recommendations({
       "metadata": None,
       "duplicates": type("D", (), {"total_duplicates": 0})(),
       "infinite": type("I", (), {})(),
       "missing": type("M", (), {"total_missing": 0, "column_reports": []})(),
       "outliers": [],
       "feature_profiles": [],
       "correlation_pairs": [],
       "target_profile": type("T", (), {"is_regression": False, "is_binary": True})(),
   })
