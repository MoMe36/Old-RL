# Reinforcement Learning for Robotics

This repo contains efforts to accomplish robotic tasks using RL. 

## Multi-Joint robotic arm  

The world designed for this task is simple: The effector of the
robotic arm has to reach a target. Inspired from OpenAI Gym, it has only a few methods: 

* `env.reset()` :  to reset the world
* `env.step(action)`: interaction with world  
* `env.initRender()` and `env.render()`:  visualize the actual setup

### State 

Currently, the state of the environement consists of a 4 dimensional vector describing the x-axis and y-axis distance between the effector and the target in the following form: [x-axis positive distance, x-axis negative distance, y-axis positive distance, y-axis negative distance]

Hence, if the vector is [-0.3,0.2] the state will be [0,0.3,0.2,0]

### Actions

You have 2*n actions availables, where n is the number of joints. You can control the number of joints, and their length


### Reward 

Tried to keep it as simple as possible (though it doesn't seem to converge for now): Reward is -1 all the time. +10 if the effector touches the target.

## RL algorithms 

Currently, the repo only has two methods: 
* DQN
* PG

However, soon Actor-Critic and other algorithms shall be added.


## Dependencies

* PyTorch for the learning part
* Numpy 
* PyGame
* Matplotlib


