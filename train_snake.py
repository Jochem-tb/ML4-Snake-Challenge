import os
import numpy as np
from stable_baselines3 import PPO, DQN
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from snake_env import Snake6x6Env

class SnakeLengthCallback(BaseCallback):
    """
    Custom callback to log the actual Snake Length (Mean and Max) to TensorBoard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_lengths = []

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if done and "snake_length" in info:
                self.episode_lengths.append(info["snake_length"])

        # Every 2000 steps, flush mean and max snake length to TensorBoard
        if self.n_calls % 2000 == 0 and len(self.episode_lengths) > 0:
            self.logger.record("snake/mean_length", np.mean(self.episode_lengths[-100:]))
            self.logger.record("snake/max_length", np.max(self.episode_lengths[-100:]))
        return True

def make_env(rank, seed=42):
    def _init():
        env = Snake6x6Env(grid_size=6)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    # Choose algorithm: "DQN", "PPO", or "MaskablePPO"
    ALGO = "MaskablePPO"
    TOTAL_TIMESTEPS = 2_000_000  # Start with 5M (~3 mins); scale to 20M for final model
    NUM_CPU = 20                # Parallel environments

    log_dir = "./snake_tb_logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Parallelize across CPU cores
    vec_env = SubprocVecEnv([make_env(i) for i in range(NUM_CPU)])

    policy_kwargs = dict(net_arch=[128, 128])

    if ALGO == "DQN":
        model = DQN(
            "MlpPolicy",
            vec_env,
            learning_rate=5e-4,
            buffer_size=100_000,
            learning_starts=5_000,
            batch_size=512,
            gamma=0.98,
            exploration_fraction=0.25,
            exploration_final_eps=0.02,
            target_update_interval=1000,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            verbose=1,
        )
    elif ALGO == "PPO":
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=256,
            n_epochs=10,
            gamma=0.98,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            verbose=1,
        )
    elif ALGO == "MaskablePPO":
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=256,
            n_epochs=10,
            gamma=0.985,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            verbose=1,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // NUM_CPU, 1),
        save_path=f"./models/{ALGO}/",
        name_prefix="snake_6x6",
    )
    length_cb = SnakeLengthCallback()

    print(f"Starting {ALGO} training for {TOTAL_TIMESTEPS:,} steps across {NUM_CPU} cores...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, length_cb],
        tb_log_name=ALGO,
    )

    model.save(f"best_{ALGO}_snake6x6")
    vec_env.close()
    print(f"Training complete! Saved as best_{ALGO}_snake6x6.zip")