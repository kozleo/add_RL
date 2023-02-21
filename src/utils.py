import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
import sys
from gymnasium import Env, spaces
from typing import List, Optional


def save_frames_as_gif(
    frames,
    path="/om2/user/leokoz8/code/add_RL/results/animations/",
    filename="gym_animation.gif",
):
    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis("off")
    num_frames = len(frames)

    def animate(i):
        # if i%(0.2*num_frames) == 0:
        print(f"Remaining number of frames : {num_frames-i}")
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=num_frames, interval=50)
    # anim.save(path + filename, fps=60,writer="imagemagick")
    writer = animation.PillowWriter(fps=5, metadata=dict(artist="Me"), bitrate=200)
    anim.save(path + filename, writer=writer)


"""
class AddictionEnv(Env):
    def __init__(self, render_mode: Optional[str] = None):
        self.observation_space = 6
        self.action_space = 3

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a

        if self.render_mode == "human":
            self.render()

        return (int(s), r, t, False, {"prob": p})

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1}
"""
