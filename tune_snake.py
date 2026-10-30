import optuna
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from train_snake import make_env
from snake_env import Snake6x6Env

def optimize_snake(trial: optuna.Trial):
    # 1. Sample hyperparameters from search space
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.94, 0.995)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.05, log=True)
    n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024])
    net_width = trial.suggest_categorical("net_width", [64, 128, 256])

    vec_env = SubprocVecEnv([make_env(i) for i in range(4)])

    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        gamma=gamma,
        ent_coef=ent_coef,
        n_steps=n_steps,
        batch_size=256,
        policy_kwargs=dict(net_arch=[net_width, net_width]),
        verbose=0,
    )

    # Fast proxy training (300,000 steps takes ~15 seconds)
    model.learn(total_timesteps=300_000)
    vec_env.close()

    # Evaluate average snake length across 30 test games
    eval_env = Snake6x6Env(grid_size=6)
    lengths = []
    for ep in range(30):
        obs, info = eval_env.reset(seed=100 + ep)
        done = False
        while not done:
            masks = eval_env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, _, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
        lengths.append(info["snake_length"])

    mean_length = float(np.mean(lengths))
    print(f"Trial {trial.number} | Mean Snake Length: {mean_length:.2f} / 36 | Params: {trial.params}")
    return mean_length

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_snake, n_trials=15)

    print("\n=== BEST HYPERPARAMETERS FOUND ===")
    print(f"Best Mean Snake Length (at 300k steps): {study.best_value:.2f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")