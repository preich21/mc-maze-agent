# Minecraft Maze Agent - Patrick's Version

A simple reinforcement learning agent working together with the mc-maze-agent-state-extractor-mod to solve a maze in Minecraft.


## Prerequisites

Create a Python `venv`, activate it and install the required dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```


## Running the Agent

First start up the Minecraft instance with the mod.
See [the Minecraft mod repo](https://github.com/preich21/mc-maze-agent-state-extractor-mod/blob/maze/README.md) for 
instructions on how to set that up.

Then run the trained agent with:

```bash
python play.py --env maze --model-path runs/ppo_minecraft/ppo_minecraft_goal.zip
```

`ppo_minecraft_goal.zip` is the latest trained model included in this repo.
It manages to solve 5x5 mazes in almost every episode.


## Training the Agent

First start up the Minecraft instances with the mod.
See [the Minecraft mod repo](https://github.com/preich21/mc-maze-agent-state-extractor-mod/blob/maze/README.md) for 
instructions on how to set that up.

Note that by default we expect 5 instances to be starte with WebSockets listening on ports `8081` - `8085`.
You can configure that by removing the addresses from the `URIS` array in `train.py`. 

Then run the training with:

```bash
python train.py --env maze
```

Note that the training will load a previously trained model from `runs/ppo_minecraft/ppo_minecraft_goal.zip` if it exists.
If you want to start training from scratch, delete or rename that file first.


## Code Structure

The following are the most relevant files of this branch:

- `play.py` and `train.py` specify the Stable Baselines3 algorithm and its hyperparameters.
- `observation.py` and `observation_vectorizer.py` define the observation space and how we convert the Minecraft state into observations.
- `action.py` defines the action space and how we convert agent actions into Minecraft actions.
- `maze_exploring_reward.py` contains the heart of the agent: the reward shaping.
