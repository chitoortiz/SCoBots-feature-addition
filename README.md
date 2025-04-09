# ReadMe for SCoBots Extension
## Setup
Perform the following commands inside of the project folder (and virtual environment if desired) for complete executability
```bash
pip install -r requirements.txt
```


## Extented collection of agents
All agents need to be stored accordingly in ```resources/checkpoints```.

All newly trained agents and some more can be downloaded via the following command executed inside of the repositories folder:
```bash
mkdir -p resources/checkpoints && curl -L "https://hessenbox.tu-darmstadt.de/getlink/fiRmSTLCkMww8nQMd3naPnm6/checkpoints.zip" -o resources/checkpoints/temp.zip && unzip -o resources/checkpoints/temp.zip -d resources/checkpoints && rm resources/checkpoints/temp.zip
```
 
These agents are trained for games of an unreleased OCATARI version. Therefore if these agents shall be evaluated the following commands are necessary to ensure the correct ALE are available:
```bash
git clone --branch develop https://github.com/k4ntz/OC_Atari
cd OC_Atari
pip install -e .
```

Example command to watch RoboTank:
```bash
python render_agent.py -g Robotank -s 0 -r env
```

## Creating executable python files from viper trees
Run the ```viper_to_trees.py``` file to create an executable python file for each viper tree existing in ```resources/viper_extracts/extract_output```.
Those can be obtained by running the viper extraction as detailed in the original readMe of the SCoBots project, and being placed at the correct location. The command for this is e.g.:
```bash
wget https://hessenbox.tu-darmstadt.de/getlink/fiRmSTLCkMww8nQMd3naPnm6/checkpoints.zip
unzip resources_seed0.zip
```

If checkpoints/viper extractions are available just run
```bash
python create_exec_code.py
```

To execute one of these files with the visualization of the current tree decisions, just execute one of the created files inside of the corresponding folder in ```resources/checkpoints```.

The game speed of the renderer for these files has been reduced to make the if-clauses readable. This is artificial through manual delay and can be changed in the renderer file if desired.

## Using the configurations
To use the SCoBots framework with configurations, just fill in the desired values/parameters inside of the yaml file and execute
```bash
python scobots_runner.py
```

if multiple instances with e.g. different seeds shall be execute use "loop" version of both


## Usage of HackAtari
HackAtari is integrated as an optional switch. the only necessary thing is to enable it via a flag in the command line, or in the configurations file.
To view and edit the mods, refer to the targeted game inside of ```scobi/environments/hackatari/games```

One example usage is running any Freeway agent with manipulated cars:
```bash
python render_agent.py -g Freeway -r env -s 0 -p default --hackatari -mods speed1
```

## Inspect the custom Seaquest Agent
To see the Seaquest agent being able to refill oxygen run the following commands. The installation of the specific OCAtari verison is necessary because this agent was trained on the released version of OCAtari and executing it on the develop branch version might cause errors:
```bash
pip install ocatari==2.0.0
```

Afterwards the agent can simply be viewed via:
```bash
python render_agent.py -g Seaquest -r env -s 0 -p default
```

To compare it to a "regular" seaquest agent just execute the following command:
```bash
python render_agent.py -g Seaquest -r env -s 0 -p default -n 0
```
