"""Peer-state prediction for anticipatory coordination under delay (P3).

Baselines (the gate): constant-velocity, Kalman, linear. The learned graph
predictor is the stretch and lives in ``graph_predictor`` (torch-only).
"""
from gossamer.prediction.base import PeerHistory, PeerPredictor, STATE_DIM
from gossamer.prediction.baselines import (
    PREDICTORS,
    ConstantVelocityPredictor,
    KalmanPredictor,
    LinearPredictor,
)

__all__ = [
    "PREDICTORS",
    "ConstantVelocityPredictor",
    "KalmanPredictor",
    "LinearPredictor",
    "PeerHistory",
    "PeerPredictor",
    "STATE_DIM",
]
