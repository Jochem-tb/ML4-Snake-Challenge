import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

class Snake6x6Env(gym.Env):
    """
    Strictly Markovian 6x6 Single-Player Snake Environment.
    Compatible with DQN, PPO, and sb3_contrib.MaskablePPO.
    """
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 8}

    def __init__(self, grid_size=6, render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode

        # Actions: 0 = Up, 1 = Right, 2 = Down, 3 = Left
        self.action_space = spaces.Discrete(4)

        # Observation: Flattened 3-channel 6x6 grid (3 * 6 * 6 = 108 floats in [0.0, 1.0])
        # Flattening works seamlessly with SB3's fast MlpPolicy
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3 * grid_size * grid_size,), dtype=np.float32
        )

        self.directions = {
            0: (-1, 0),  # Up
            1: (0, 1),   # Right
            2: (1, 0),   # Down
            3: (0, -1),  # Left
        }

        # Hunger clock: max steps allowed without eating before truncation (prevents infinite loops)
        self.max_starvation_steps = grid_size * grid_size * 2  # 72 steps

        self.snake = None
        self.direction = None
        self.food = None
        self.steps_since_food = 0
        self.total_steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Start snake of length 3 in the middle of the 6x6 board moving Right
        mid = self.grid_size // 2
        self.snake = deque([(mid, mid), (mid, mid - 1), (mid, mid - 2)])
        self.direction = 1  # Right
        self.steps_since_food = 0
        self.total_steps = 0

        self._spawn_food()
        return self._get_obs(), {"snake_length": len(self.snake)}

    def _spawn_food(self):
        empty_cells = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in self.snake
        ]
        if not empty_cells:
            self.food = None  # Board is 100% full (Max possible length 36 achieved!)
        else:
            idx = self.np_random.integers(0, len(empty_cells))
            self.food = empty_cells[idx]

    def _get_obs(self):
        grid = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)

        # Channel 0: Head position
        head_r, head_c = self.snake[0]
        grid[0, head_r, head_c] = 1.0

        # Channel 1: Markovian Body Gradient (Neck = highest, Tail = lowest)
        # Tells the network exactly which segment vacates first
        body_len = len(self.snake)
        for i, (r, c) in enumerate(list(self.snake)[1:]):
            grid[1, r, c] = (body_len - 1 - i) / max(1, (body_len - 1))

        # Channel 2: Food position
        if self.food is not None:
            grid[2, self.food[0], self.food[1]] = 1.0

        return grid.flatten()

    def action_masks(self) -> np.ndarray:
        """
        Used by MaskablePPO to forbid instantaneous 180-degree neck reversals.
        """
        mask = np.ones(4, dtype=bool)
        opposite_direction = (self.direction + 2) % 4
        mask[opposite_direction] = False
        return mask

    def step(self, action):
        action = int(action)
        # Prevent 180-degree instantaneous self-collision if standard DQN/PPO tries it
        if action == (self.direction + 2) % 4 and len(self.snake) > 1:
            action = self.direction
        self.direction = action

        dr, dc = self.directions[self.direction]
        head_r, head_c = self.snake[0]
        new_head = (head_r + dr, head_c + dc)

        self.steps_since_food += 1
        self.total_steps += 1

        terminated = False
        truncated = False
        reward = 0.0

        # Check wall collision
        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            terminated = True
            reward = -1.0
            return self._get_obs(), reward, terminated, truncated, {"snake_length": len(self.snake)}

        # Check eating food
        ate_food = (new_head == self.food)

        # Note: Moving into the current tail tip is SAFE if the snake does NOT eat food,
        # because the tail tip vacates on this exact tick!
        body_to_check = list(self.snake) if ate_food else list(self.snake)[:-1]
        if new_head in body_to_check:
            terminated = True
            reward = -1.0
            return self._get_obs(), reward, terminated, truncated, {"snake_length": len(self.snake)}

        # Move snake forward
        self.snake.appendleft(new_head)

        if ate_food:
            # Progressive reward scaling: eating at longer lengths is rewarded slightly higher
            reward = 1.0 + (len(self.snake) * 0.05)
            self.steps_since_food = 0
            self._spawn_food()
            if self.food is None:
                # Won the entire 6x6 board (Length = 36)!
                reward += 10.0
                terminated = True
        else:
            self.snake.pop()
            # Tiny step penalty encourages taking the shortest safe path to food
            reward = -0.005

        # Hunger clock truncation (Workshop 1B TimeLimit protection)
        if self.steps_since_food >= self.max_starvation_steps:
            truncated = True
            reward -= 0.5

        info = {"snake_length": len(self.snake)}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        board = np.full((self.grid_size, self.grid_size), ".", dtype=str)
        for r, c in list(self.snake)[1:]:
            board[r, c] = "o"
        hr, hc = self.snake[0]
        board[hr, hc] = "H"
        if self.food is not None:
            board[self.food[0], self.food[1]] = "F"
        grid_str = "\n".join(" ".join(row) for row in board)
        print(f"\nLength: {len(self.snake)} | Steps: {self.total_steps}\n{grid_str}")