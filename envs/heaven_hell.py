import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from tkinter import *
import time


class HeavenHellEnv(gym.Env):
    def __init__(self, grid_len_x=5, grid_len_y=7, seed=0, mdp=False, rendering=False):

        self.visualize = rendering
        self.mdp = mdp

        if self.visualize:
            # Create top-level window
            self.root = Tk()
            self.root.title("Heaven-Hell")

            # Create canvas to hold grid world
            self.canvas = Canvas(self.root, width="500", height="500")
            self.canvas.grid(column=0, row=0, sticky=(N, W, E, S))

        # Create grid world
        self.grid_len_x = grid_len_x
        self.grid_len_y = grid_len_y
        state_mat = np.ones((self.grid_len_x, self.grid_len_y))
        state_mat[1:self.grid_len_x, :self.grid_len_y//2] = 0
        state_mat[1:self.grid_len_x-1, self.grid_len_y//2+1:] = 0
        self.state_mat = state_mat

        # Determine pixel length of each block
        num_col = state_mat.shape[1]
        pixel_width = 480
        while pixel_width % num_col != 0:
            pixel_width -= 1

        num_row = state_mat.shape[0]

        block_length = pixel_width / num_col

        self._state = 25

        if self.visualize:
            # Create rectangles
            for i in range(num_row):
                for j in range(num_col):
                    x_1 = self.grid_len_x + block_length * j
                    y_1 = self.grid_len_y + block_length * i
                    x_2 = x_1 + block_length
                    y_2 = y_1 + block_length

                    if self.state_mat[i][j] == 1:
                        color = "white"
                    else:
                        color = "grey"

                    self.canvas.create_rectangle(x_1, y_1, x_2, y_2, fill=color)
        self.info_coords = [[self.grid_len_x - 1, self.grid_len_y - 1]]
        self.info_cells = []
        for coord in self.info_coords:
            coord_x = coord[0]
            coord_y = coord[1]
            self.info_cells.append(coord_x * self.grid_len_y + coord_y)

        # color directional area
        if self.visualize:
            self._color(self.info_coords, 'blue')
            self.root.update()

        self.action_space = spaces.Discrete(4)

        self.low_state = np.array(
            [0.0, 0.0, -1.0], dtype=np.float32  # normalized x, y, direction
        )
        self.high_state = np.array(
            [1.0, 1.0, 1.0], dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )

        self.state_space = gym.spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )

        self.goal_loc = None
        self.hell_loc = None
        self.enter_info = False
        self.true_state = None
        self.use_image = False

        self.step_reward = 0.0

    def _convert_coordinate(self, coord_x, coord_y):
        """Convert a (x, y) point to a 1D number

        Args:
            coord_x (int): x coordinate (from 0, horizontal)
            coord_y (int): y coordinate (from 0, vertical)
        """
        return coord_x * self.grid_len_y + coord_y + 1

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """an environment step function

        Args:
            action (int): 0: left, 1: up, 2: right, 3: down
        """
        old_state = np.copy(self._state)
        old_state = int(old_state)

        done = False
        reward = self.step_reward
        enter_info_area = False

        self._state = self._get_next_state(self.state_mat, np.copy(old_state), action)

        # Clear the previous position
        if self._state != old_state and self.visualize:
            if old_state not in self.info_cells:
                self.canvas.itemconfig(old_state + 1, fill="white")
            else:
                self.canvas.itemconfig(old_state + 1, fill="blue")

        if self._state != self.goal_loc:
            if self.visualize:
                self.canvas.itemconfig(self._state + 1, fill="black")
        else:
            # Reached the goal
            if self.visualize:
                self.canvas.itemconfig(self._state + 1, fill="orange")
            done = True
            reward += 1.0

        if self._state != self.hell_loc:
            if self.visualize:
                self.canvas.itemconfig(self._state + 1, fill="black")
        else:
            # Reached hell
            if self.visualize:
                self.canvas.itemconfig(self._state + 1, fill="red")
            done = True
            reward += -1.0

        if self._state in self.info_cells:
            self.enter_info = True
            if self.visualize:
                self.canvas.itemconfig(self._state + 1, fill="black")
            enter_info_area = True
        else:
            if self.visualize:
                self._color(self.info_coords, 'blue')

        if self.visualize:
            self.root.update()

        obs = self._prepare_obs(self._state,
                                self.goal_loc if enter_info_area else None)
        self._prepare_state(self._state)

        if self.visualize:
            time.sleep(0.1)

        info = {}
        info["success"] = reward > self.step_reward

        if self.mdp:
            obs = self.get_state()

        return obs, reward, done, info

    def render(self, mode='human'):
        pass

    def reset(self):
        # Clear the previous goal coors
        if self.goal_loc is not None and self.visualize:
            self.canvas.itemconfig(self.goal_loc + 1, fill="white")
            self.canvas.itemconfig(self._state + 1, fill="white")

        self.enter_info = False

        if np.random.rand() < 0.5:
            self.goal_loc = 0
            self.hell_loc = self.grid_len_y - 1
        else:
            self.goal_loc = self.grid_len_y - 1
            self.hell_loc = 0

        # Hightlight the chosen goal area
        if self.visualize:
            self.canvas.itemconfig(self.goal_loc + 1, fill="green")
            self.canvas.itemconfig(self.hell_loc + 1, fill="red")

        self._state = (self.grid_len_x * self.grid_len_y) - 1 - self.grid_len_y - self.grid_len_y // 2

        if self.visualize:
            # Hightlight the agent
            self.canvas.itemconfig(self._state + 1, fill="black")

            # Highlight info area
            self._color(self.info_coords, 'blue')

            self.root.update()

        self._prepare_state(self._state)
        if self.mdp:
            return self.get_state()
        return self._prepare_obs(self._state)

    def _prepare_obs(self, state, goal=None):
        """observation = (pos_x, pos_y, direction to goal)

        Args:
            state (int): agent position
            goal (int, optional): goal position. Defaults to None.
        """
        direction = 0.0

        if goal is not None:
            xg, yg = self._state_to_xy(self.goal_loc)
            if xg > 0:
                direction = 1.0  # right
            else:
                direction = -1.0  # left

        x, y = self._state_to_xy(state)
        return np.array([x/(self.grid_len_y - 1), y/(self.grid_len_x - 1), direction])

    def get_state(self):
        return self.true_state.copy()

    def _prepare_state(self, state):
        """state = (pos_x, pos_y, true direction to goal)

        Args:
            state (int): agent position
            goal (int, optional): goal position. Defaults to None.
        """
        xg, yg = self._state_to_xy(self.goal_loc)
        if xg > 0:
            true_direction = 1.0  # right
        else:
            true_direction = -1.0  # left
        x, y = self._state_to_xy(state)
        self.true_state = np.array([x/(self.grid_len_y - 1),
                                    y/(self.grid_len_x - 1),
                                    true_direction])

    def _state_to_xy(self, state):
        """convert a state to x y coordinate (the origin is in the top left corner)

        Args:
            state (int): index

        Returns:
            List: [x, y]
        """
        x = state % self.grid_len_y
        y = (state - x) // self.grid_len_y

        return [x, y]

    def _color_goal_area(self, goal_loc, color='green'):
        """color goal area

        Args:
            goal_loc (List):
            color (str):
        """
        if self.visualize:
            self.canvas.itemconfig(self._convert_coordinate(goal_loc[0], goal_loc[1]), fill=color)

    def _get_next_state(self, state_mat, state, action):

        num_col = state_mat.shape[1]
        num_row = state_mat.shape[0]

        state_row = int(state/num_col)
        state_col = state % num_col

        # If action is "left"
        if action == 2:
            if state_col != 0 and state_mat[state_row][state_col - 1] == 1:
                # print("Moving Left")
                state -= 1
        # If action is "up"
        elif action == 1:
            if state_row != 0 and state_mat[state_row - 1][state_col] == 1:
                # print("Moving Up")
                state -= num_col
        # If action is "right"
        elif action == 0:
            if state_col != (num_col - 1) and state_mat[state_row][state_col + 1] == 1:
                # print("Moving Right")
                state += 1
        # If action is "down"
        else:
            if state_row != (num_row - 1) and state_mat[state_row + 1][state_col] == 1:
                # print("Moving Down")
                state += num_col

        return state

    def _color(self, coordinates, color):
        """Color given coordinates

        Args:
            coordinates (List): a list holding coordinates to color
            color (str): color to fill
        """
        for coord in coordinates:
            self.canvas.itemconfig(self._convert_coordinate(coord[0], coord[1]), fill=color)

class HeavenHellEnvV1(HeavenHellEnv):
    def __init__(self, grid_len_x=6, grid_len_y=9, seed=0, rendering=False):
        super().__init__(grid_len_x=grid_len_x, grid_len_y=grid_len_y, seed=seed, rendering=rendering)

if __name__ == "__main__":
    from getkey import getkey, keys
    env = HeavenHellEnv(rendering=True)
    # env = HeavenHellEnvV1(rendering=True)
    obs = env.reset()
    print("Reset: ", obs, env.get_state())

    for _ in range(100):

        while True:

            key = getkey()

            if key == keys.LEFT:
                action = 2
                break

            if key == keys.RIGHT:
                action = 0
                break

            if key == keys.UP:
                action = 1
                break

            if key == keys.DOWN:
                action = 3
                break

        obs, reward, done, info = env.step(action)
        print(obs, env.get_state(), reward, done, info)
        print(env._state)

        if done:
            obs = env.reset()
            print("Reset: ", obs, env.get_state())