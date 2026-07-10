"""Shared helpers for working with recommendation outputs.

This module keeps small cross-cutting utilities out of the recommendation
engine so presentation layers can normalize heterogeneous recommendation
shapes without depending on private implementation details.
"""

from __future__ import annotations

from typing import Any, List

from preml.schema import Recommendation


def normalize_recommendation_items(items: Any) -> List[Recommendation]:
    """Return recommendation-like items as a flat list of Recommendation objects."""
    if items is None:
        return []
    if isinstance(items, Recommendation):
        return [items]
    if isinstance(items, list):
        return [item for item in items if isinstance(item, Recommendation)]
    if hasattr(items, "action") and hasattr(items, "category"):
        return [items]
    return []