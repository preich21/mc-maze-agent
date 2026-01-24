# Minecraft Maze Agent

A simple reinforcement learning agent working together with the mc-maze-agent-state-extractor-mod to solve a maze in Minecraft.

## Project Structure

After setting up the initial code base together, we divided our efforts into two different paths:
1. Patrick developed maze-specific logic and trained a model for it on branch [maze](https://github.com/preich21/mc-maze-agent/tree/maze)
2. Axel trained models on much simpler tasks on branch [feat/multiple-start-points](https://github.com/preich21/mc-maze-agent/tree/feat/multiple-start-points)

For further information, please see the READMEs on the respective branches.

## Run it!

Note that to run anything, you need to switch the branch to [maze](https://github.com/preich21/mc-maze-agent/tree/maze) or [feat/multiple-start-points](https://github.com/preich21/mc-maze-agent/tree/feat/multiple-start-points).

### Installation
```shell
# create venv first
python -m venv
# then install required packages
pip install -r requirements.txt 
```

### Run training

```shell
python train.py --env <simple/maze>
```

### Use the resulting model

```shell
python play.py --env <simple/maze> --model <id>
```