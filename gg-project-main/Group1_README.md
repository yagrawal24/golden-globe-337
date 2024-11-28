# Group 1 Golden Globes Project

This file contains some information regarding our project submission so you can navigate our file more easily.

## Setup

Run setup for environment.yml file.

Please also download en_core_web_sm using the following:
- python -m spacy download en_core_web_sm

## Code

The project utilizes gg_api.py to run the autograder. For our code, gg_api.py calls functions from helper_functions.py, which contains the key functionality for our project.

## Running our Code

The human-readable file is being generated from a function call in "main" of gg_api.py. In order to reduce clutter, our human-readable file runs the same functionality that is run in the autograder.

*Of Note:* For new award name datasets (i.e. the award names are different than our development data), please change the award_names variable to the corresponding award names in gg_api.py for better autograder results. WE ARE ONLY HARD CODING THIS FOR THE SAKE OF THE AUTOGRADER AND ARE PREDICTING AWARDS SEPARATELY.

## Thought Process

The thought process behind our code is documented in final_submission.ipynb. We felt this would allow us to be a bit more verbose with our rationale.

## Our Answers

Our answers for the candidates to host the show and win or present each award, along with the award names we generated, can all be found in human_readable_output.txt.