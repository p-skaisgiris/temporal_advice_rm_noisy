import random
from enum import IntEnum

import numpy as np
from gymnasium.envs.registration import register
from minigrid.core.actions import Actions
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Floor, WorldObj
from minigrid.minigrid_env import Grid, MiniGridEnv, spaces


class RockSampleEnv(MiniGridEnv):
    """
    Partially Observable Rock Sampling Environment
    The agent needs to sample good rocks and avoid bad ones
    """

    def __init__(
        self,
        width=10,
        height=7,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        num_rocks=5,
        good_rock_prob=0.5,
        step_penalty=-0.1,
        exit_reward=10.0,
        good_rock_reward=5.0,
        bad_rock_penalty=-5.0,
        sensor_use_penalty=-0.2,
        sensor_efficiency=0.5,
        max_steps=100,
        **kwargs,
    ):
        # Environment configuration
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.num_rocks = num_rocks
        self.good_rock_prob = good_rock_prob

        # Reward parameters
        self.turn_penalty = step_penalty
        self.step_penalty = step_penalty
        self.exit_reward = exit_reward
        self.good_rock_reward = good_rock_reward
        self.bad_rock_penalty = bad_rock_penalty
        self.sensor_use_penalty = sensor_use_penalty

        # Sensor properties
        self.sensor_efficiency = sensor_efficiency

        # Track rocks in the environment
        self.rocks = {}
        self.rock_positions = {}

        # Define custom actions:
        # MiniGrid default actions:
        # 0: turn left
        # 1: turn right
        # 2: move forward
        # 3: pickup
        # 4: drop
        # 5: toggle
        # 6: done

        # Custom actions:
        # 7..7+num_rocks-1: sense rock[i]
        self.sense_rock_base = 7  # First sensing action ID
        total_actions = self.sense_rock_base + num_rocks

        # Define a mission space
        mission_space = MissionSpace(mission_func=self._gen_mission)

        max_view_size = (
            width * height if width * height % 2 == 1 else width * height + 1
        )

        # Initialize the environment
        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=False,  # Agent can't see through walls
            agent_view_size=max_view_size,  # In rocksample the agent knows where the rocks are
            **kwargs,
        )

        # Set the action space
        self.action_space = spaces.Discrete(total_actions)

        self.letter_types = ["left", "right", "forward", "pickup", "sense"]

    @staticmethod
    def _gen_mission():
        return "Sample good rocks and exit the grid on the right side."

    def _gen_grid(self, width, height):
        """Generate the grid for the environment"""
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Create exit area on the right side
        for j in range(1, height - 1):
            self.grid.set(width - 1, j, None)

        # Reset rock tracking
        self.rocks = {}
        self.rock_positions = {}

        # Place rocks
        self._place_rocks()

        # Place the agent
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        # Set the mission
        self.mission = self._gen_mission()

    def _gen_rock_quality(self, prob: float) -> bool:
        return random.random() < prob

    def _place_rocks(self):
        """Place rocks randomly in the grid"""
        # Generate random positions for rocks
        positions = []
        for _ in range(self.num_rocks):
            while True:
                # Don't place rocks on the right edge (exit area) or on the agent start position
                pos = (
                    self._rand_int(1, self.width - 2),
                    self._rand_int(1, self.height - 1),
                )
                if pos not in positions and pos != tuple(self.agent_start_pos):
                    positions.append(pos)
                    break

        # Place the rocks
        for i, pos in enumerate(positions):
            is_good = self._gen_rock_quality(self.good_rock_prob)

            # rock = Floor("green") if is_good else Floor("red")
            rock = Floor("grey")
            rock.is_good = is_good  # Add attribute to track rock quality

            self.grid.set(pos[0], pos[1], rock)
            self.rocks[i] = rock
            self.rock_positions[i] = pos

    def _get_rock_at_agent_pos(self):
        """Check if there's a rock at the agent's position"""
        for rock_id, pos in self.rock_positions.items():
            if pos == (self.agent_pos[0], self.agent_pos[1]):
                return self.rocks[rock_id], rock_id
        return None, None

    def _sample_rock(self):
        """
        Sample the rock at the agent's current position
        Returns the reward based on whether the rock is good or bad
        """
        pos = self.agent_pos

        # Check if the agent is on a rock
        for rock_id, rock_pos in list(self.rock_positions.items()):
            if rock_pos == (pos[0], pos[1]):
                rock = self.rocks[rock_id]
                # Remove the rock after sampling
                self.grid.set(pos[0], pos[1], None)
                del self.rock_positions[rock_id]
                del self.rocks[rock_id]

                # Return reward based on rock status
                if rock.is_good:
                    return self.good_rock_reward, True, rock_id
                else:
                    return self.bad_rock_penalty, False, rock_id

        # No rock at this position
        return self.bad_rock_penalty, None, None

    def _sense_rock(self, rock_id):
        """
        Sense the status of a specific rock
        The accuracy decreases exponentially with distance
        Returns an observation and the sensor use penalty
        """
        if rock_id not in self.rocks:
            # Rock doesn't exist
            return None, self.sensor_use_penalty

        rock = self.rocks[rock_id]
        rock_pos = self.rock_positions[rock_id]

        # Calculate distance from agent to rock
        distance = np.sqrt(
            (self.agent_pos[0] - rock_pos[0]) ** 2
            + (self.agent_pos[1] - rock_pos[1]) ** 2
        )

        # Calculate accuracy based on distance
        accuracy = np.exp(-distance * self.sensor_efficiency)

        # Generate noisy observation
        correct_obs = rock.is_good
        if self.np_random.random() < accuracy:
            # Correct observation
            rock.color = "green"
            obs = correct_obs
        else:
            # Incorrect observation
            rock.color = "red"
            obs = not correct_obs

        return obs, self.sensor_use_penalty

    def reset(self, **kwargs):
        """Reset the environment"""
        return super().reset(**kwargs)

    def step(self, action):
        """
        Take an action in the environment
        Returns observation, reward, terminated, truncated, info
        """
        self.step_count += 1
        reward = self.step_penalty  # Base penalty for each step
        terminated = False
        truncated = self.step_count >= self.max_steps
        rock_observation = None
        rock_id = None
        sampled_rock_status = None

        # Process different actions
        if action == Actions.left:  # Turn left
            reward += self.turn_penalty
            self.agent_dir = (self.agent_dir - 1) % 4

        elif action == Actions.right:  # Turn right
            reward += self.turn_penalty
            self.agent_dir = (self.agent_dir + 1) % 4

        elif action == Actions.forward:  # Move forward
            # Get the cell in front of the agent
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)

            # Move if the cell is empty or can be overlapped
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos

                # Check if agent is at exit
                if self.agent_pos[0] == self.width - 1:
                    reward += self.exit_reward
                    terminated = True
                else:
                    reward += self.step_penalty

        elif action == Actions.pickup:  # Sample rock (using pickup)
            sample_reward, sampled_rock_status, rock_id = self._sample_rock()
            reward += sample_reward
            # self.events +=

        # elif action == Actions.done:  # Done action - check if at exit
        #     if self.agent_pos[0] == self.width - 1:
        #         reward += self.exit_reward
        #         terminated = True

        elif action >= self.sense_rock_base:  # Sense rock
            rock_id = action - self.sense_rock_base
            rock_observation, sensor_penalty = self._sense_rock(rock_id)
            reward += sensor_penalty

        # Generate observation
        obs = self.gen_obs()

        # Create info dictionary
        info = {
            "rock_observation": rock_observation,
            "rock_id": rock_id,
            "sampled_rock_status": sampled_rock_status,
        }

        # Add exit event
        if self.agent_pos[0] == self.width - 1:
            info["exit"] = True

        return obs, reward, terminated, truncated, info

    def seed(self, seed):
        random.seed(seed)

    def get_events(self):
        """Get current events"""
        # events = ""
        # # Check if at exit
        # if self.agent_pos[0] == self.width - 1:
        #     events += "e"  # Exit

        # # Check if on rock
        # rock, rock_id = self._get_rock_at_agent_pos()
        # if rock is not None:
        #     events += f"r{rock_id}"  # At rock position

        # return self.events
        return ""

    def get_propositions(self):
        """Get all possible propositions in the environment"""
        return self.letter_types


# Register the environment
register(
    id="RockSample-v0",
    entry_point="envs.minigrid.rocksample:RockSampleEnv",
)
