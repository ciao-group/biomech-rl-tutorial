# MyoSuite Finger Pose Tutorial

This project provides a simple tutorial for setting up, customizing, training, and visualizing a musculoskeletal finger model using [MyoSuite](https://github.com/MyoHub/myosuite), [Gymnasium](https://gymnasium.farama.org/), and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

## Google Colab
 
You can use the provided Jupyter Notebook (`Colab_Tutorial.ipynb`) 
1. Clone the entire project directory into your Google Colab session.
2. Open `Colab_Tutorial.ipynb` in Colab.
3. Run the cells sequentially!

## Project Structure

* **`tutorial.py`**: A basic script to load the `myoFingerPoseFixed-v0` environment and apply a simple oscillating sine wave to the muscle activations. Shows how to interact with the environment without RL.
* **`reward_tutorial.py`**: Defines a custom environment class (`CustomRewardPoseEnv`) that inherits from MyoSuite's `PoseEnvV0`. It demonstrates how to override the reward function and inject custom rewards (e.g., an "efficiency" reward).
* **`train.py`**: Trains a Proximal Policy Optimization (PPO) agent using Stable-Baselines3 on the custom environment defined in `reward_tutorial.py`.
* **`visualize.py`**: Loads the trained PPO model, runs it in the custom environment, and renders the episode. The rendered frames are saved as an MP4 video in the `video/` directory.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) as its package manager for extremely fast dependency resolution and environment isolation.

1. Install `uv` if you haven't already:
```bash
pip install uv
```

2. Sync the environment and install dependencies defined in `pyproject.toml` (which includes `myosuite`, `gymnasium`, `stable-baselines3`, and `imageio`):
```bash
uv sync
```

Alternatively, you can manually run a script using uv to handle the environment automatically:
```bash
uv run python tutorial.py
```

## Workflow: Human-in-the-Loop Reward Tuning

The setup is designed to allow you to easily modify what the agent "cares about" and immediately see the results.

### 1. Modify the Reward
Open `reward_tutorial.py`. Inside the `__init__` method, you can adjust the `default_weights` or modify the `custom_reward_weights` passed in `train.py`. You can also inject entirely new reward logic inside the `get_reward_dict` function.

### 2. Train the Agent
Run the training script to train a new agent based on your customized reward logic:
```bash
uv run python train.py
```
This will train the agent for 100,000 timesteps across 10 parallel environments and save the model to `models/ppo_myofinger_custom_reward.zip`.

### 3. Visualize the Results
Once training is complete, run the visualization script to watch the agent's behavior:
```bash
uv run python visualize.py
```
This will run the trained policy and render an off-screen video. The resulting video will be saved as `video/episode_custom_reward.mp4`. Open this file to see how your reward changes affected the finger's movement!

## Note on Rendering
MyoSuite handles rendering through the MuJoCo physics engine natively (`env.mj_renderer.render_offscreen`). The `visualize.py` script automatically manages fetching these frames and compiling them into a video using `imageio`.