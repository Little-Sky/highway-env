from typing import Tuple

from gym.envs.registration import register
from gym import GoalEnv
import numpy as np
from numpy.core._multiarray_umath import ndarray

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.parking_env import ParkingEnv
from highway_env.road.lane import StraightLane, LineType
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.objects import Landmark

class TestInheritParkingEnv(ParkingEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({})
        return config

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.vehicle = self.action_type.vehicle_class(self.road, [0, 0], 2*np.pi*self.np_random.rand(), 0)
        self.road.vehicles.append(self.vehicle)

        lane = self.np_random.choice(self.road.network.lanes_list())
        self.goal = Landmark(self.road, lane.position(lane.length/2, 0), heading=lane.heading)
        self.road.objects.append(self.goal)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> float:
        """
        Proximity to the goal is rewarded
        We use a weighted p-norm
        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        return -np.power(np.dot(np.abs(achieved_goal - desired_goal), self.REWARD_WEIGHTS), p)

    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type.observe()
        return self.compute_reward(obs['achieved_goal'], obs['desired_goal'], {})

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return self.compute_reward(achieved_goal, desired_goal, {}) > -self.SUCCESS_GOAL_REWARD

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the goal is reached."""
        obs = self.observation_type.observe()
        return self.vehicle.crashed or self._is_success(obs['achieved_goal'], obs['desired_goal'])

class MultiGoalsParkingEnv(ParkingEnv):
    """
    A continuous control environment.
    It implements a reach-type task, where the agent observes their position and speed and must
    control their acceleration and steering so as to reach any given goals.
    """
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "num_goals": 5
        })
        return config

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.vehicle = self.action_type.vehicle_class(self.road, [0, 0], 2*np.pi*self.np_random.rand(), 0)
        self.road.vehicles.append(self.vehicle)
        lanes = self.np_random.choice(self.road.network.lanes_list(), self.config['num_goals'], replace=False)
        self.goals = [Landmark(self.road, lane.position(lane.length/2, 0), heading=lane.heading) for lane in lanes]
        self.road.objects.append(self.goals)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> np.ndarray:
        """
        Proximity to the goal is rewarded
        We use a weighted p-norm
        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        min_dist = float("inf")
        for goal : desired_goal:
            min_dist = min(min_dist, np.power(np.dot(np.abs(achieved_goal - desired_goal), self.REWARD_WEIGHTS), p))
        return -min_dist

    def _reward(self, action: np.ndarray) -> np.ndarray:
        obs = self.observation_type.observe()
        return self.compute_reward(obs['achieved_goal'], obs['desired_goal'], {})

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return np.max(self.compute_reward(achieved_goal, desired_goal, {})) > -self.SUCCESS_GOAL_REWARD

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the goal is reached."""
        obs = self.observation_type.observe()
        return self.vehicle.crashed or self._is_success(obs['achieved_goal'], obs['desired_goal'])

register(
    id='testInheritParking-v0',
    entry_point='highway_env.envs:TestInheritParkingEnv',
    max_episode_steps=100
)


register(
    id='multiGoalsParking-v0',
    entry_point='highway_env.envs:MultiGoalsParkingEnv',
    max_episode_steps=100
)
