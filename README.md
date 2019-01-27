# John Conway's Game of Life & Reinforcement Learning
![](https://upload.wikimedia.org/wikipedia/en/d/d0/Game_of_life_animated_glider_2.gif)

## About

Project's description here soon.

### Modules structure
##### evn
There is a straightforward python implementation that works for the first experiments:  [env](https://github.com/denistr16/GoL-RL/blob/master/env/env_naive_sphere.py)

##### loss
There is a first loss that defines as:   L1 distance between the amount of all cells and all living cells
[loss](https://github.com/denistr16/GoL-RL/blob/master/loss/losses.py)

##### model
We already have a naive linear implementation of the first agent:
[linear_observer_planter](https://github.com/denistr16/GoL-RL/blob/master/model/linear_observer_planter.py)

##### patterns
There are a few hardcoded patterns that you can play with - [file](https://github.com/denistr16/GoL-RL/blob/master/patterns/gliders.py)

## Install
For the moment - just clone the repo

## Usage
Latest examples of usage:
- Run environment: [Env run](https://github.com/denistr16/GoL-RL/blob/master/envs_run_template_02.ipynb)
- Run agent:  [Env run](https://github.com/denistr16/GoL-RL/blob/master/agent_run_template_02.ipynb)

## How to commit

