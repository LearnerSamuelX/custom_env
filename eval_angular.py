import warnings

# Suppress AVX2 warning
warnings.filterwarnings("ignore", message="Your system is avx2 capable but pygame was not built with support for it")

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import time

gym.envs.register(
    id="NewEnvBipedal_ang-V0",
    entry_point="custom_env.bipedal_walker_angular:BipedalWalker",
    max_episode_steps=1000,
    reward_threshold=300.0,
)


def make_env():
    return gym.make("NewEnvBipedal_ang-V0", render_mode="human")


env = make_env()

env = make_vec_env(lambda: env)

# Load the trained model

model_path = os.path.join("training", "saved_models", "NewBipedalWalker_final_custom_no_stop_thres_h8k12")
model = PPO.load(model_path, env=env)

# Evaluate the trained model
start_time = time.time()
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1, render=True)
end_time = time.time()
trip_duration = end_time - start_time
print(f"Mean reward: {mean_reward} ± {std_reward}")
print(f"Trip duration is: {trip_duration}")


env.close()
