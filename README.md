# John Conway's Game of Life & Reinforcement Learning
![](https://upload.wikimedia.org/wikipedia/en/d/d0/Game_of_life_animated_glider_2.gif)

## About

The project is a mix of Reinforcement Learning concept and famous
John Conway's Game of Life, a cellular automaton that generates thrilling
structures that very looks like real living cells and complex molecules.
We assume that such an environment could be an interesting source of
new features as well as an inspiration itself. So.. The major goals are
the experiments on SOTA in RL  with the following transfer to
real-world applications.

### Modules
 - [Environments](#env)
 - [Models](#model)
 - [Losses](#loss)
 - [Patterns](#patterns)

##### Env
There are we have two envs:
 - An implementation GoL with classical rules:  [env](https://github.com/denistr16/GoL-RL/blob/master/env/env_naive_torus.py)
 - An env implementation for two players: [env](https://github.com/denistr16/GoL-RL/blob/master/env/env_2players_naive_torus.py)

##### Loss
There is a first loss we designed. It defines as:
L1 distance between the amount of living cells and all the field cells:
[loss](https://github.com/denistr16/GoL-RL/blob/master/loss/losses.py)

##### Model
There are three models:
- Random: [random](https://github.com/denistr16/GoL-RL/blob/master/model/random_model.py)
- Linear: [linear_observer_planter](https://github.com/denistr16/GoL-RL/blob/master/model/linear_observer_planter.py)
- Actor-Critic model: [a2c](https://github.com/denistr16/GoL-RL/blob/master/model/a2c.py)


##### Patterns
There are a few hardcoded patterns that you can play with - [gliders](https://github.com/denistr16/GoL-RL/blob/master/patterns/gliders.py)

## Install
For the moment - just clone the repo and run the latest notebooks

## Usage
Latest examples of usage:
- Run environment: [Env](https://github.com/denistr16/GoL-RL/blob/master/envs_run_template_02.ipynb)
- Run agent:  [Agent](https://github.com/denistr16/GoL-RL/blob/master/agent_run_template_04.ipynb)

## Experiment analysis
TBA

## How to commit
We are small at the moment and don't have any special requirements.
We will be appreciated if you get one of the issues and help us.
Just remember: new issue = new branch -> pull request.
