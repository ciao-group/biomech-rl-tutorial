import collections
import os
import myosuite
import numpy as np
from myosuite.envs.myo.myobase.pose_v0 import PoseEnvV0


class CustomRewardPoseEnv(PoseEnvV0):
    """
    A custom wrapper/child class around MyoSuite's PoseEnvV0.
    This lets us intercept and modify the reward logic and its weighting
    while retaining the physics and action space of the original environment.
    """

    def __init__(self, *args, **kwargs):
        # Do not pop custom arguments. Let them pass through to the parent 
        # so they can be pickled for multiprocessing.
        self.use_muscle_noise = kwargs.get('use_muscle_noise', False)
        self.custom_reward_weights = kwargs.get('custom_reward_weights', None)

        # IMPORTANT: MyoSuite's base environment calls `step()` during its initialization.
        # Therefore, any attributes needed by our custom `step()` (like `self.rng`) 
        # must be initialized BEFORE calling `super().__init__`.
        if self.use_muscle_noise:
            self.rng = np.random.default_rng()
            self.sigma_signal_dependent = 0.103 * 1.8
            self.sigma_constant = 0.185 * 2.5

        # Call the parent constructor with all kwargs, including our custom ones.
        super().__init__(*args, **kwargs)

        # --- DEFAULT WEIGHTS (Matching Original Env) ---
        # We keep a local copy here so it's easy to see and modify the defaults 
        # if you don't pass `custom_reward_weights`.
        default_weights = {
            "pose": 1.0,
            "bonus": 4.0,
            "act_reg": 1.0,
            "penalty": 50,
            "efficiency": 0.0  # Custom weight disabled by default
        }

        # Apply the default weights
        self.rwd_keys_wt.update(default_weights)

        # If the user passed custom weights, overwrite the defaults
        if self.custom_reward_weights is not None:
            self.rwd_keys_wt.update(self.custom_reward_weights)
    
    def apply_motor_noise(self, action):
        """
        Applies signal-dependent and constant noise to the action.
        """
        signal_dependent_noise = self.rng.lognormal(
            mean=0.0,
            sigma=self.sigma_signal_dependent,
            size=action.shape
        ) - 1.0

        constant_noise = self.rng.normal(
            loc=0.0,
            scale=self.sigma_constant,
            size=action.shape
        )

        noisy_action = (1 + signal_dependent_noise) * action + constant_noise
        return np.clip(noisy_action, -1.0, 1.0)

    def step(self, a, **kwargs):
        if self.use_muscle_noise:
            a = self.apply_motor_noise(a)
        return super().step(a, **kwargs)

    def get_reward_dict(self, obs_dict):
        """
        Overriding the core reward calculation function.
        It must return a dictionary containing "dense", "sparse", "solved", and "done".
        """
        # The goal distance (Euclidean distance between current pose and target pose)
        pose_dist = np.linalg.norm(obs_dict["pose_err"], axis=-1)

        # Muscle activation magnitude (how hard the muscles are firing)
        act_mag = np.linalg.norm(self.obs_dict["act"], axis=-1)
        if self.mj_model.na != 0:
            act_mag = act_mag / self.mj_model.na

        far_th = 4 * np.pi / 2  # Hardcoded distance threshold for "failing" the task

        # ==========================================================
        # YOUR CUSTOM REWARD COMPONENT
        # Let's say we want to heavily reward the agent if the pose is correct,
        # AND it is using almost NO muscle activation (efficiency).
        is_close = pose_dist < self.pose_thd
        is_efficient = act_mag < 0.05


        efficiency_reward = 1.0 * np.logical_and(is_close, is_efficient)
        # ==========================================================

        # Build the reward dictionary
        rwd_dict = collections.OrderedDict((
            # Optional Keys (Weighted)
            ("pose", -1.0 * pose_dist),
            ("bonus", 1.0 * (pose_dist < self.pose_thd) + 1.0 * (pose_dist < 1.5 * self.pose_thd)),
            ("penalty", -1.0 * (pose_dist > far_th)),
            ("act_reg", -1.0 * act_mag),

            # Our custom key
            ("efficiency", efficiency_reward),


            ("sparse", -1.0 * pose_dist),
            ("solved", pose_dist < self.pose_thd),
            ("done", pose_dist > far_th),
        ))

        # The 'dense' key must contain the weighted sum of the components.
        # This is the actual single float value returned by `env.step()`.
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items() if key in rwd_dict], axis=0
        )

        return rwd_dict


def main():
    myosuite_dir = os.path.dirname(myosuite.__file__)
    model_path = os.path.join(myosuite_dir, "simhive", "myo_sim", "finger", "myofinger_v0.xml")

    print("Initializing Custom Environment...")
    env = CustomRewardPoseEnv(
        model_path=model_path,
        target_jnt_range={
            "IFadb": (0, 0),
            "IFmcp": (0, 0),
            "IFpip": (0.75, 0.75),
            "IFdip": (0.75, 0.75),
        },
        viz_site_targets=("IFtip",),
        normalize_act=True,
        use_muscle_noise=True,
        custom_reward_weights={
            "pose": 2.0,
            "bonus": 10.0,
            "act_reg": 0.1,
            "penalty": 50.0,
            "efficiency": 2.0
        }
    )

    env.reset()

    print("\nStarting an episode to test the new rewards...")
    total_reward = 0
    for i in range(100):
        # We sample a completely random muscle activation
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Every 20 steps, let's print the detailed breakdown of the rewards
        if i % 20 == 0:
            print(f"\nStep {i}:")
            print(f"  Dense Reward (total): {reward:.4f}")
            print(f"  Detailed Breakdown:   {info['rwd_dict']}")

        if terminated or truncated:
            break

    env.close()
    print(f"\nEpisode finished. Total dense reward: {total_reward:.2f}")


if __name__ == "__main__":
    main()