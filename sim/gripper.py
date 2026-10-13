"""One candidate motor-angle coordinate for both command and feedback.

The affine transmission is NOT measured. Keeping its existing feedback
endpoints preserves recorded-pose geometry while fixing command/state algebra.
The mirrored finger-joint gain follows virtual work, not a fitted multiplier.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GripperMap:
    open_motor: float
    closed_motor: float
    open_joint: float
    joint_unit: str = 'rad'

    def __post_init__(self):
        if (not np.isfinite([self.open_motor, self.closed_motor, self.open_joint]).all()
                or self.open_motor >= self.closed_motor or self.open_joint == 0
                or self.joint_unit not in ('rad', 'm')):
            raise ValueError('invalid candidate gripper transmission')

    @property
    def joint_per_motor(self):
        return -self.open_joint / (self.closed_motor - self.open_motor)

    def to_joint(self, motor):
        return (np.asarray(motor) - self.closed_motor) * self.joint_per_motor

    def to_motor(self, joint):
        return self.closed_motor + np.asarray(joint) / self.joint_per_motor

    def joint_gain(self, motor_gain):
        return motor_gain / self.joint_per_motor**2

    def motor_torque(self, joint_torque):
        return np.asarray(joint_torque) * self.joint_per_motor

    def describe(self):
        return {'schema': 'shared-motor-angle/1', 'motor_unit': 'rad',
                'open_motor_rad': self.open_motor, 'closed_motor_rad': self.closed_motor,
                'open_joint_value': self.open_joint, 'joint_unit': self.joint_unit,
                'joint_per_motor': self.joint_per_motor,
                'physically_calibrated': False,
                'note': 'Same affine map for motor target and feedback; endpoints remain data-derived candidates.'}
