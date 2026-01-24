# Minecraft Maze Agent

A simple reinforcement learning agent working together with the mc-maze-agent-state-extractor-mod to solve a maze in Minecraft.

## Running the agent

First start up the Minecraft instance with the mod.
See [the Minecraft mod repo](https://github.com/preich21/mc-maze-agent-state-extractor-mod/blob/maze/README.md) for 
instructions on how to set that up.

Then run the trained agent with:

```bash
python play.py --env maze --model-path runs/ppo_minecraft/ppo_minecraft_goal.zip
```

## Training the agent

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
