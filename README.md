# Inference with Speechbrain
A repository made for performing inference with [Speechbrain](https://speechbrain.github.io/)

## Overview
This repository gives an easy way to try out the pre-entrained models for speech recognition, speaker identification and speech enhancement. It can also help to fine tune by using the pretrained model as component. I used it to learn how to use all of it's usage.

## Installation
### Clone the repository
```sh
git clone https://github.com/MathisLB/speechbrain_inference.git
```
### Set up the environment
Follow the requirements of the [Speechbrain repository](https://github.com/speechbrain/speechbrain) for that.

## Usage
Each of the python program work on it's own except the one for fine tuning which must be used in this order :
1. pipeline_preppin.py
2. fine_tuning.py