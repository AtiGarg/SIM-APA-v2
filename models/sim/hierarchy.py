"""
Class-ID based semantic hierarchy for SIM-APA v2.0.

The hierarchy uses COCO class IDs from YOLOv10.

Tier 1:
    person, bicycle, motorcycle

Tier 2:
    car, bus, truck

Tier 3:
    traffic light, stop sign

Tier 4:
    background / other classes
"""

from __future__ import annotations


# COCO class IDs used by YOLO
SEMANTIC_CLASS_WEIGHTS = {
    0: 4.0,   # person
    1: 4.0,   # bicycle
    3: 4.0,   # motorcycle

    2: 3.0,   # car
    5: 3.0,   # bus
    7: 3.0,   # truck

    9: 2.0,   # traffic light
    11: 2.0,  # stop sign
}


def get_class_weight(class_id: int, default_weight: float = 1.0) -> float:
    """
    Return the semantic importance weight for a YOLO class ID.

    Parameters
    ----------
    class_id : int
        COCO class ID predicted by YOLO.
    default_weight : float
        Weight assigned to classes not explicitly included in the hierarchy.

    Returns
    -------
    float
        Semantic importance weight.
    """
    return SEMANTIC_CLASS_WEIGHTS.get(int(class_id), default_weight)
