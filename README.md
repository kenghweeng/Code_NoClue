Project: Code No Clue
==============================
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/kenghweeng/code_noclue/presentation_env/JSS/app.py)

Please visit our deployed application here, for a detailed explanation of the hospital scheduling problem as well as the solutioning methodology.

This proposed approach was largely based on "A Reinforcement Learning Environment For Job-Shop Scheduling". The optimized environment is available as a separate [repository](https://github.com/prosysscience/JSSEnv)

We cite the original [paper](https://arxiv.org/abs/2104.03760) as follows:

```
@misc{tassel2021reinforcement,
      title={A Reinforcement Learning Environment For Job-Shop Scheduling}, 
      author={Pierre Tassel and Martin Gebser and Konstantin Schekotihin},
      year={2021},
      eprint={2104.03760},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

Getting Started
------------

**This code has been tested on Ubuntu 18.04 and MacOs 10.15, with Python 3.8.
Some users have reported difficulties running this program on Windows.**

The code uses mainly Ray's RLLib, Tensorflow and Wandb. Note that it is necessary to sign up for a Weight and Bias account, if you want your metrics logged.
Otherwise, just remove all occurrence of wandb and log the metrics in another way. Make sure you have `git`, `cmake`, `zlib1g`, and, on Linux, `zlib1g-dev` installed.

To get started with setting up the dev environment, you may want to use a `virtualenv`, or something more refined like `Poetry` or `pipenv` and proceed with:
```shell (bash/zsh/sh)
git clone https://github.com/kenghweeng/Code_NoClue.git
git checkout presentation_env
pip3 install /JSS/requirements.txt
```

**Important: Your instance must follow [standard specifications](http://jobshop.jjvh.nl/explanation.php#taillard_def).** We also provide a comprehensive explanation of the specification in the application above.

Project Organization
------------
    **root**
    ├── README.md                 <- The top-level README for developers using this project.
    ├── requirements.txt            <- for reproducibility of environment
    └── JSS
        ├── dispatching_rules/            <- Contains the code to run traditional heuristics FIFO and MWR
        ├── env/                          <- a modified OpenGym AI environment based off the above cited paper
        ├── instances/                    <- Contains benchmark Taillard's & COVID simulations
        ├── images/                       <- Contains the static images as well as GIFs
        ├── schedules/                    <- Contains CSVs detailing patient treatment allocation chronologically
        ├── solutions/                    <- Seralized pickles containing exact solution sequences and makespan
        ├── app.py                        <- Contains the source code for the Streamlit application.
        ├── covid_instance_generator.py   <- Realistic Discrete event simulation of COVID patient treatments.
        ├── CP.py                         <- Google based OR-Tool's cp model for the JSS problem. (For fun)
        ├── CustomCallbacks.py            <- A special RLLib's callback used to save the best solution found.
        ├── default_config.py             <- default config used for the disptaching rules.
        ├── generate_gantt.py             <- Used for generating images/GIFs, solutions, as well as schedules.
        ├── main.py                       <- PPO approach, the main file to call to reproduce our approach.
        └── models.py                     <- Tensorflow model who mask logits of illegal actions.
        
--------

### User Tasks: These tasks assume the proper installation of requirements.txt was completed.

1. Generating COVID patient treatment demands (i.e scheduling problems). In the project root,
```
for i in {1..1}; do python -m JSS.covid_instance_generator $i 2 covid2_$i; done
for i in {1..100}; do python -m JSS.covid_instance_generator $i 5 covid5_$i; done
for i in {1..100}; do python -m JSS.covid_instance_generator $i 8 covid8_$i; done
for i in {1..100}; do python -m JSS.covid_instance_generator $i 10 covid10_$i; done
for i in {1..100}; do python -m JSS.covid_instance_generator $i 15 covid15_$i; done
for i in {1..100}; do python -m JSS.covid_instance_generator $i 20 covid20_$i; done
```


2. Generating solutions for the simulations created from Point 1: 
* Solving the scheduling problem with the FIFO heuristic (Assuming you are in project root):
```
for i in {1..100}; do python -W ignore -m JSS.dispatching_rules.FIFO covid2_$i; done
for i in {1..100}; do python -W ignore -m JSS.dispatching_rules.FIFO  covid5_$i; done
for i in {1..100}; do python -W ignore -m JSS.dispatching_rules.FIFO  covid8_$i; done
for i in {1..100}; do python -W ignore -m JSS.dispatching_rules.FIFO  covid10_$i; done
for i in {1..100}; do python -W ignore -m JSS.dispatching_rules.FIFO  covid15_$i; done
for i in {1..100}; do python -W ignore -m JSS.dispatching_rules.FIFO  covid20_$i; done
```
* Solving the scheduling problem with Google OR tools: (Assuming you are in project root):
```
for i in {1..100}; do python -W ignore -m JSS.CP  covid2_$i; done
for i in {1..100}; do python -W ignore -m JSS.CP  covid5_$i; done
for i in {1..100}; do python -W ignore -m JSS.CP  covid8_$i; done
for i in {1..100}; do python -W ignore -m JSS.CP  covid10_$i; done
for i in {1..100}; do python -W ignore -m JSS.CP  covid15_$i; done
for i in {1..100}; do python -W ignore -m JSS.CP  covid20_$i; done
```
* Solving the scheduling problem with the PPO reinforcement learning approach:
```
cd JSS
python3.8 main.py <instance_name, for example "covid2_1">
```
The above code would generate a seralized pickle to the `solutions/` folder, where we are able to obtain the scheduled timings and also the total makespan for our RL agent. You may want to open the pickled file to take a look.

Also, we have set the training episode here to be for 10 minutes. Some things you may want to change here can be:
- Line 149, change the quantity to the amount you want to train in seconds.
- Lines 26, you may want to change the visible GPUs to those GPU cards which are available for training!


3. Suppose you want to plot out the images (GIFs) and a nicely formatted schedule in CSV, we have also provided a generate_gantt.chart for your use! We assume you are currently in the `JSS` folder, which you should be at, when running the `main.py` for the reinforcement learning agent training.

`python3.8 generate_gantt.py <instance_name, for example "covid2_1">`

The script will essentially make use of the instance specifications found in the `instances/` folder, as well as the generated solution pickle in `solutions/`, to generate the relevant GIFs, and timetables in the `images/` and `schedules/` folders respectively.

