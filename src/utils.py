import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
import sys
from gymnasium import Env, spaces
from typing import List, Optional
import numpy as np

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


def update_Q_function(Q,s,s_prime,a,r,gamma,alpha):

    delta = r + gamma*np.max(Q[s_prime,:]) - Q[s,a]
    Q[s,a] += alpha*delta
    
    return Q

def update_Q_and_E(Q,E,s,s_prime,a,r,gamma,alpha,lam,d):
    
    # Update eligibility trace and Q-function
    delta = reward + gamma * Q[next_state, next_action] - Q[state, action]
    E[state, action] += 1
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            Q[s, a] += alpha * delta * E[s, a]
            E[s, a] = gamma * lambd * E[s, a]

    return Q,E

def epsilon_greedy_policy(s,Q,epsilon,num_actions):
    if np.random.uniform(0,1) > epsilon:
        a = np.argmax(Q[s,:])            
    else:
        a = np.random.choice(np.arange(num_actions))
        
    return a

def step(s,a):
    
    if s == 0 and a == 0:
        drug_press = True
    else:
        drug_press = False
    
    s_prime = 1           
        
    return s_prime, drug_press
        


def dopamine_release(t, release_type = 'quick_rise',k = 1,tau = 1,sigma = 1):
    if release_type == 'quick_rise':
        return np.maximum(0,k*tau*t*np.exp(-tau*t))
    if release_type == 'gauss':
        return np.maximum(0,k*np.exp(-(tau*t/sigma)**2))


from scipy.special import softmax               

def v_hat(w,S):
    return np.dot(w,S)    
    
def update_actor_critic(S,a,z_w,z_theta,w,theta,I,R):

    lam_theta = 0.9
    lam_w = 0.9
    
    alpha_theta = 0.5
    alpha_w = 0.5

    gamma = 0.9

    delta = R  + gamma*v_hat(w,S) - v_hat(w,S)

    z_w = gamma*lam_w*z_w + S


    one_hot_action = np.zeros(len(z_theta))
    one_hot_action[a] = 1
    z_theta = gamma*lam_theta*z_theta + I*(one_hot_action - softmax(theta))

    w = w + alpha_w*delta*z_w
    theta = theta + alpha_theta*delta*z_theta

    I = gamma*I

    return z_w,z_theta,w,theta,I



    