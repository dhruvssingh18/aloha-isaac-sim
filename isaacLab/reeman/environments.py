
# scripts/environment.py
import numpy as np
import random

class Environment:
    """
    Placeholder IsaacLab environment for SAC training.
    Simulates a Reeman-style robot with lidar, imu, pose, and joints.
    """

    def __init__(self, config):
        self.n_lidar = config.get('n_lidar', 64)
        self.max_lidar_range = config.get('max_lidar_range', 10.0)
        self.step_count = 0
        self.max_steps = config.get('max_steps', 100)
        self.domain_params = None
        self.state = None

    def sample_domain_randomization(self):
        self.domain_params = {
            "mass_mult": random.uniform(0.8, 1.2),
            "wheel_friction": random.uniform(0.6, 1.4),
            "motor_gain": random.uniform(0.9, 1.1),
            "lidar_noise": random.uniform(0.0, 0.05),
            "imu_noise": random.uniform(0.0, 0.1),
            "wheel_base": random.uniform(0.28, 0.35),
            "max_wheel_speed": random.uniform(3.0, 6.0),
            "collision_threshold": random.uniform(0.15, 0.5),
        }

    def reset(self):
        # Reset environment and return initial observation
        self.step_count = 0
        self.sample_domain_randomization()
        self.state = {
            "pose": np.array([0.0, 0.0, 0.0]),         # x, y, yaw
            "vel": np.array([0.0, 0.0]),               # linear, angular
            "joints": np.array([0.0, 0.0]),            # left, right wheels
            "lidar": np.ones(self.n_lidar) * self.max_lidar_range,
            "imu": np.zeros(3)
        }
        return self.get_observation()

    def step(self, action):
        # Apply action to robot, update simulated state
        self.step_count += 1
        left_cmd, right_cmd = action
        max_wheel_speed = self.domain_params["max_wheel_speed"]

        left_speed = left_cmd * max_wheel_speed
        right_speed = right_cmd * max_wheel_speed

        x, y, yaw = self.state["pose"]
        linear_vel = 0.5 * (left_speed + right_speed)
        angular_vel = (right_speed - left_speed) / (self.domain_params["wheel_base"] + 1e-9)
        dt = 0.05

        x += linear_vel * np.cos(yaw) * dt
        y += linear_vel * np.sin(yaw) * dt
        yaw += angular_vel * dt

        self.state["pose"] = np.array([x, y, yaw])
        self.state["vel"] = np.array([linear_vel, angular_vel])
        self.state["joints"] = np.array([left_speed, right_speed])

        # Add noise
        lidar = self.state["lidar"] + np.random.normal(0, self.domain_params["lidar_noise"], self.n_lidar)
        lidar = np.clip(lidar, 0, self.max_lidar_range) / self.max_lidar_range
        imu = self.state["imu"] + np.random.normal(0, self.domain_params["imu_noise"], 3)

        self.state["lidar"] = lidar
        self.state["imu"] = imu

        obs = self.get_observation()
        reward = linear_vel * dt                   # simple forward reward
        done = lidar.min() < self.domain_params["collision_threshold"] or self.step_count >= self.max_steps
        info = {}

        return obs, reward, done, info

    def get_observation(self):
        # Concatenate lidar, imu, pose+vel, joints
        obs = np.concatenate([
            self.state["lidar"].astype(np.float32),
            self.state["imu"].astype(np.float32),
            np.array([*self.state["pose"], *self.state["vel"]], dtype=np.float32),
            self.state["joints"].astype(np.float32)
        ])
        return obs
