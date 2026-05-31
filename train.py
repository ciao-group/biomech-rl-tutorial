import os
import myosuite
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.wrappers import TimeLimit
from reward_tutorial import CustomRewardPoseEnv

def make_custom_env(**kwargs):
    env = CustomRewardPoseEnv(**kwargs)
    return TimeLimit(env, max_episode_steps=100)


def train():
    # Use myosuite.__file__ to get the exact path to the package directory
    myosuite_dir = os.path.dirname(myosuite.__file__)
    model_path = os.path.join(myosuite_dir, "simhive", "myo_sim", "finger", "myofinger_v0.xml")

    # Define the arguments for the environment, which will be passed to each parallel process
    env_kwargs = {
        "model_path": model_path,
        "target_jnt_range": {
            "IFadb": (0, 0),
            "IFmcp": (0, 0),
            "IFpip": (0.75, 0.75),
            "IFdip": (0.75, 0.75),
        },
        "viz_site_targets": ("IFtip",),
        "normalize_act": True,
        "use_muscle_noise": True,
        "custom_reward_weights": {
            "pose": 1.0,
            "bonus": 4.0,
            "act_reg": 1.0,
            "penalty": 50,
        }
    }

    # Create a vectorized environment that runs 10 instances of the environment in parallel
    n_envs = 10
    
    # We pass our factory function `make_custom_env`. 
    env = make_vec_env(
        make_custom_env,
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=env_kwargs
    )

    # Save a checkpoint every 10000 steps
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./logs/',
                                             name_prefix='myo_finger_custom_reward')

    model = PPO("MlpPolicy", env, verbose=1)

    print(f"Starting training on {n_envs} parallel custom reward environments...")
    # Train for 100_000 timesteps. Note: total_timesteps is distributed across all envs.
    model.learn(total_timesteps=100_000, callback=checkpoint_callback)

    # Save the final model
    os.makedirs('models', exist_ok=True)
    model.save("models/ppo_myofinger_custom_reward")
    print("Training finished and model saved to 'models/ppo_myofinger_custom_reward.zip'.")


if __name__ == "__main__":
    train()