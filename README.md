# Minecraft Maze Agent - Axel Version

A simple reinforcement learning agent working together with the mc-maze-agent-state-extractor-mod to solve a maze in Minecraft.

## Results

I added various git tags for different trained models:
 - `2_simple_world_1_first_working_model`
 - `15_streaming`
 - `21_history_length_1`

It's required to use the version of the minecraft mod of the equally name git tag 
because the message format etc. changed over time. All resulting models are contained in the [models folder](./models) 
and somewhat named to a reflect what I tried to change in this run. Note that most of the models cannot be run with the current
code of the env, because observation and action spaces changed over time.

## Environment Description
The central entry point is our `MinecraftEnv` in [env.py](mc_env/env.py). It implements the OpenAI Gym interface and sets up the
Websocket connection to the Minecraft Mod. Also, it parses the incoming observations and flattens them to vectors. It also
creates logical objects out of the vectorized action it receives.

[simple_goal_reward.py](wrappers/simple_goal_reward.py) is a wrapper around the env that implements the reward shaping.

[action.py](mc_env/action.py) contains the logic to convert between logical actions (like "move forward", "turn left") and vectors.

[observation_vectorizer.py](wrappers/observation_vectorizer.py) contains the logic to convert the incoming observations from the Minecraft mod into flat vectors.

[observations_buffer.py](mc_env/observations_buffer.py) buffers the incoming observations to decouple receiving observations from processing them.

[action_history.py](mc_env/action_history.py) keeps track of the last N actions taken by the agent and appends them to the observation vector.

[debug_observation.py](wrappers/debug_observation.py) is a wrapper adds logging information about the observations for debugging purposes.
[debug_tensorboard.py](callbacks/debug_tensorboard.py) adds logging information added in the env or wrappers around it to Tensorboard for debugging purposes.

## Run it!

### Installation
```shell
# create venv first
python -m venv
# then install required packages
pip install -r requirements.txt 
```

### Run training

```shell
python train.py --env simple --time mid
```

### Use the resulting model

```shell
python play.py --env simple --model 27
```