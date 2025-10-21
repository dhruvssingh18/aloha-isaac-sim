import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
import random, time

class ReemanManipEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.render = render
        if render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("r2d2.urdf", [0, 0, 0.1])

        self.left_joint, self.right_joint = 2, 3

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self._spawn_object()

    def _spawn_object(self):
        # Randomize object type, size, and friction (domain randomization)
        size = random.uniform(0.05, 0.2)
        friction = random.uniform(0.3, 1.0)
        xpos, ypos = random.uniform(0.5, 1.5), random.uniform(-0.5, 0.5)
        self.obj = p.loadURDF("cube_small.urdf", [xpos, ypos, size / 2],
                              globalScaling=size / 0.1)
        p.changeDynamics(self.obj, -1, lateralFriction=friction)

    def _randomize_environment(self):
        # Randomize lighting, texture, physics parameters
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, int(random.choice([0, 1])))
        p.setGravity(0, 0, -9.8 * random.uniform(0.8, 1.2))

    def reset(self, seed=None, options=None):
        p.resetBasePositionAndOrientation(self.robot, [0, 0, 0.1], [0, 0, 0, 1])
        self._randomize_environment()
        self._spawn_object()
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
        obj_pos, _ = p.getBasePositionAndOrientation(self.obj)
        distance = np.linalg.norm(np.array(robot_pos[:2]) - np.array(obj_pos[:2]))

        obs = np.array([
            *robot_pos[:2],
            *obj_pos[:2],
            distance,
            np.clip(distance / 2.0, 0, 1),
            random.random(), random.random(), random.random()  # pseudo-sensor readings
        ], dtype=np.float32)
        return obs

    def step(self, action):
        left_vel, right_vel = action * 5.0
        p.setJointMotorControl2(self.robot, self.left_joint, p.VELOCITY_CONTROL, targetVelocity=left_vel)
        p.setJointMotorControl2(self.robot, self.right_joint, p.VELOCITY_CONTROL, targetVelocity=right_vel)
        p.stepSimulation()
        if self.render: time.sleep(1/240)

        obs = self._get_obs()
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
        obj_pos, _ = p.getBasePositionAndOrientation(self.obj)
        distance = np.linalg.norm(np.array(robot_pos[:2]) - np.array(obj_pos[:2]))

        # Reward for pushing object closer to target zone
        reward = -distance
        done = distance < 0.2 or self.steps > 500
        self.steps += 1
        return obs, reward, done, False, {}

