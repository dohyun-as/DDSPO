#!/bin/bash
# This script is used to generate captions for the images in the specified directory.
cd prompts_generation
python create_prompts.py --o ../data/captions/test --n 1000