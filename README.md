# iao-spotify-good4

📊 Data
[Kaggle Dataset:  500K+ Spotify Songs with Lyrics,Emotions & More](https://www.kaggle.com/datasets/devdope/900k-spotify/data?select=spotify_dataset.csv)

- Import the file `spotify_dataset.csv` into the data folder. 

- Run the script `clean-good4.py` to obtain the cleansed csv


🛠️ Python Environment Setup

This project uses a reproducible Conda environment defined in environment.yml.
To create and activate the environment, run the following commands:

1. Install Conda (if needed)

You can use Anaconda or Miniconda:
https://docs.conda.io/en/latest/miniconda.html

2. Create the environment

From the root of the repository:

conda env create -f environment.yml


This will install all required dependencies with correct versions.

3. Activate the environment
conda activate <env_name>


The environment name is defined inside environment.yml (the name: field).
Example: if the file says name: harmi-env, then run:

conda activate harmi-env

4.