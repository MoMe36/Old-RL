# Reinforcement Learning for Robotics

This repo contains efforts to accomplish robotic tasks using RL. 

## Multi-Joint robotic arm  

The world designed for this task is simple: The effector of the
robotic arm has to reach a target. Inspired from OpenAI Gym, it has only a few methods: 

* `env.reset()` :  to reset the world
* `env.step(action)`: interaction with world  
* `env.initRender()` and `env.render()`:  visualize the actual setup

### Actions

You have 2*n actions availables, where n is the number of joints. You can control the number of joints, and their length

## RL algorithms 

Currently, the repo only has one method: 
* DQN

However, soon PG and other algorithms shall be added.


## Dependencies

* PyTorch for the learning part
* Numpy 
* PyGame
* Matplotlib


