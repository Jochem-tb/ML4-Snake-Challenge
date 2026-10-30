import time
import numpy as np
import pygame
from sb3_contrib import MaskablePPO
from snake_env import Snake6x6Env

def evaluate_and_watch(model_path="best_MaskablePPO_snake6x6", num_eval_episodes=100, watch=True):
    model_path = "./models/MaskablePPO/snake_6x6_1600000_steps.zip"
    env = Snake6x6Env(grid_size=6)
    model = MaskablePPO.load(model_path)

    # 1. Statistical Evaluation across 100 episodes
    lengths = []
    perfect_games = 0
    for ep in range(num_eval_episodes):
        obs, info = env.reset(seed=1000 + ep)
        done = False
        while not done:
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        lengths.append(info["snake_length"])
        if info["snake_length"] == 36:
            perfect_games += 1

    print(f"=== Evaluation over {num_eval_episodes} Episodes ===")
    print(f"Mean Length : {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
    print(f"Max Length  : {np.max(lengths)} / 36")
    print(f"Perfect 36s : {perfect_games} ({perfect_games/num_eval_episodes*100:.1f}%)")

    if not watch:
        return

    # 2. Live Pygame Visualization
    pygame.init()
    cell_size = 80
    screen_size = env.grid_size * cell_size
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("ML4 6x6 RL Snake Live Demo")

    obs, info = env.reset()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        masks = env.action_masks()
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # Render board
        screen.fill((30, 30, 30))
        # Draw Food
        if env.food is not None:
            fr, fc = env.food
            pygame.draw.rect(screen, (220, 50, 50), (fc * cell_size + 8, fr * cell_size + 8, cell_size - 16, cell_size - 16), border_radius=8)

        # Draw Snake with Markovian color gradient (Bright green head -> Dark green tail)
        body_len = len(env.snake)
        for i, (r, c) in enumerate(env.snake):
            intensity = int(255 - (i / max(1, body_len)) * 160)
            color = (50, intensity, 80) if i > 0 else (255, 215, 0)  # Gold head
            pygame.draw.rect(screen, color, (c * cell_size + 4, r * cell_size + 4, cell_size - 8, cell_size - 8), border_radius=6)

        pygame.display.flip()
        time.sleep(0.12)

        if terminated or truncated:
            print(f"Episode finished! Final Length: {info['snake_length']} / 36")
            time.sleep(0.8)
            obs, info = env.reset()

    pygame.quit()

if __name__ == "__main__":
    evaluate_and_watch()