import numpy as np
from scipy.spatial.transform import Rotation as R

def euler_to_quaternion(roll, pitch, yaw):
    # Create a rotation object using roll, pitch, yaw (in radians)
    # rotation = R.from_euler('zyx', [yaw, pitch, roll], degrees=False)
    rotation = R.from_euler('yzx', [pitch, yaw, roll], degrees=False)

    # Get the quaternion
    quaternion = rotation.as_quat()  # [x, y, z, w]
    
    return quaternion

# Example usage:
roll = np.radians(0)    # Roll in degrees converted to radians
pitch = np.radians(180)   # Pitch in degrees converted to radians
yaw = np.radians(90)     # Yaw in degrees converted to radians

quaternion = euler_to_quaternion(roll, pitch, yaw)
print("Quaternion (x, y, z, w):", quaternion)
