import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
import sys

def save_frames_as_gif(frames, path='/om2/user/leokoz8/code/add_RL/results/animations/', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')
    num_frames = len(frames)

    def animate(i):
        #if i%(0.2*num_frames) == 0:
        print(f"Remaining number of frames : {num_frames-i}")
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = num_frames, interval=50)
    #anim.save(path + filename, fps=60,writer="imagemagick")
    writer = animation.PillowWriter(fps=5, metadata=dict(artist='Me'), bitrate=200)
    anim.save(path + filename, writer=writer)