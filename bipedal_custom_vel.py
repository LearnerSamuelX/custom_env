import warnings
import gym

# Suppress AVX2 warning
warnings.filterwarnings("ignore", message="Your system is avx2 capable but pygame was not built with support for it")

import os
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

gym.envs.register(
    id="NewEnvBipedal_vel-V0",
    entry_point="custom_env.bipedal_walker_vel:BipedalWalker",
    max_episode_steps=1000,
    reward_threshold=300.0,
)


def make_env():
    env = gym.make("NewEnvBipedal_vel-V0")
    return env

from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold


import torch as th

# tuned hyperparamters
config = {
    "policy": "MlpPolicy",
    "n_steps": 1503,
    "batch_size": 32,
    "gamma": 0.9606344595370512,
    "learning_rate": 0.0004803031415590949,
    "ent_coef": 5.2492091292781885e-06,
    "clip_range": 0.19171786835210225,
    "n_epochs": 2,
    "gae_lambda": 0.8571941809479062,
    "max_grad_norm": 2.372533951435877,
    "vf_coef": 0.39445800472492965,
}

TIME_STEPS_VAL = int(1e6)
SUFFIX = str(TIME_STEPS_VAL)

run = wandb.init(
    project="bipedal-walker",
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
    name="sam_PPO_" + SUFFIX + "_custom_no_stop_threshold_1125e-3",
    save_code=True,
)

# Define the policy_kwargs to specify the network architecture
policy_kwargs = {
    "net_arch": {
        "pi": [256, 256],
        "vf": [256, 256],
    },
    "activation_fn": th.nn.ReLU,
}
env = make_env()
env = make_vec_env(lambda: env, n_envs=1)

# Create the PPO model
model = PPO(env=env, verbose=1, tensorboard_log=f"runs/{run.id}", policy_kwargs=policy_kwargs, **config)

# Use deterministic actions for evaluation
eval_callback = EvalCallback(
    env,
    # callback_on_new_best=callback_on_best,
    eval_freq=2000,
    deterministic=True,
    best_model_save_path="./logs/",
    verbose=1,
)

model.learn(
    total_timesteps=TIME_STEPS_VAL,
    callback=[
        WandbCallback(
            gradient_save_freq=1000,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
        eval_callback,
    ],
)


PPO_path = os.path.join("training", "saved_models", "NewBipedalWalker_final_custom_no_stop_thres_8e-2")
model.save(PPO_path)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

run.finish()
env.close()
