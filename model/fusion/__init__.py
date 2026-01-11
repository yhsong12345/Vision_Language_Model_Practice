"""
Sensor Fusion Modules

This module contains fusion strategies for multi-modal sensor data:
- SensorFusion: Simple concatenation with self-attention
- CrossModalFusion: Cross-attention between modalities
- HierarchicalFusion: Hierarchical spatial-temporal fusion
- GatedFusion: Learned gating for modality weighting
"""

from .sensor_fusion import SensorFusion, CrossModalFusion, HierarchicalFusion, GatedFusion

__all__ = [
    "SensorFusion",
    "CrossModalFusion",
    "HierarchicalFusion",
    "GatedFusion",
]
