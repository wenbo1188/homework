import os
import tensorflow as tf
from tensorflow import keras
import gym
import numpy as np
import pickle
import load_policy

GAME = "Walker2d-v2"
DAGGER = True

def main():
    model = keras.models.load_model(os.path.join("model", GAME + "-model-40000-d64-d64-e100.h5"))
    env = gym.make(GAME)
    policy_fn = load_policy.load_policy(os.path.join("experts", GAME + ".pkl"))
    returns = []
    observations = []
    actions = []
    expert_actions = []
    max_steps = env.spec.timestep_limit
    for i in range(20):
        print("interation", i)
        obs = env.reset()
        done = False
        steps = 0
        totalr = 0
        while not done:
            # print(obs.shape)
            action = model.predict(obs[None,:])
            if DAGGER:
                expert_action = policy_fn(obs[None,:])
                expert_actions.append(expert_action)
                observations.append(obs)
            # actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            # env.render()
            if steps >= max_steps:
                break

        returns.append(totalr)
    
    extra_expert_data = {"observations": np.array(observations),
                       "actions": np.array(expert_actions)}
    data = {"mean": np.mean(returns), "std": np.std(returns)}
    print("mean: {}".format(data["mean"]))
    print("std: {}".format(data["std"]))
    with open(os.path.join("expert_data", "extra_" + GAME + ".pkl"), "wb") as f:
        pickle.dump(extra_expert_data, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join("result_data", GAME + "_1"), "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()