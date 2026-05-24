"""Robotic-arm extensions for vision-guided differentiable physics."""

from .planar_arm import DifferentiablePlanarArm
from .robot_dataset import SyntheticRobotReachDataset
from .robot_policy import VisionArmPolicy

__all__ = ["DifferentiablePlanarArm", "SyntheticRobotReachDataset", "VisionArmPolicy"]
