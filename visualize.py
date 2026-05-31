import os
import myosuite
import imageio
import numpy as np
from stable_baselines3 import PPO
from reward_tutorial import CustomRewardPoseEnv
from gymnasium.wrappers import TimeLimit

def visualize_and_save_video():
    # We will use the model trained on our custom environment
    model_path = "models/ppo_myofinger_custom_reward.zip"
    video_dir = "video"

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run train.py first.")
        return

    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, "episode_custom_reward.mp4")

    # Use myosuite.__file__ to get the exact path to the package directory
    myosuite_dir = os.path.dirname(myosuite.__file__)
    model_path_env = os.path.join(myosuite_dir, "simhive", "myo_sim", "finger", "myofinger_v0.xml")

    base_env = CustomRewardPoseEnv(
        model_path=model_path_env,
        target_jnt_range={
            "IFadb": (0, 0),
            "IFmcp": (0, 0),
            "IFpip": (0.75, 0.75),
            "IFdip": (0.75, 0.75),
        },
        viz_site_targets=("IFtip",),
        normalize_act=True,
        use_muscle_noise=False
    )
    
    # We must wrap it exactly as we did in training!
    env = TimeLimit(base_env, max_episode_steps=100)

    model = PPO.load(model_path, env=env)

    obs, info = env.reset()
    frames = []

    print("Running an episode with the custom-trained agent...")

    max_steps = 200
    for _ in range(max_steps):
        try:
            # We must access the unwrapped environment's renderer
            frame = env.unwrapped.mj_renderer.render_offscreen(
                width=640,
                height=480,
                camera_id=-1
            )
            frames.append(np.asarray(frame, dtype=np.uint8))
        except Exception as e:
            env.close()
            print(f"Rendering failed: {e}")
            return

        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    env.close()

    if frames:
        print(f"Saving video to {video_path}...")
        imageio.mimsave(video_path, frames, fps=30)
        print("Video saved successfully!")
    else:
        print("Failed to capture any frames for the video.")


if __name__ == "__main__":
    visualize_and_save_video()