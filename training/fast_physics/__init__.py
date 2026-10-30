"""Fast, configurable quadrotor dynamics for robust-policy research."""

from .dynamics import FastQuadrotor
from .parameters import VehicleParameters, load_vehicle_parameters, sample_scenario
from .env import AdaptiveDroneEnv
from .system_identification import OnlineThrustEstimator
from .curriculum import randomization_for_stage
from .vehicle_family import VehicleFamilyEnv, load_vehicle_profiles, vehicle_feature_vector

__all__ = [
    "FastQuadrotor",
    "VehicleParameters",
    "load_vehicle_parameters",
    "sample_scenario",
    "AdaptiveDroneEnv",
    "OnlineThrustEstimator",
    "randomization_for_stage",
    "VehicleFamilyEnv",
    "load_vehicle_profiles",
    "vehicle_feature_vector",
]
