# iao-spotify-good4

## 📊 Data
[Kaggle Dataset:  500K+ Spotify Songs with Lyrics,Emotions & More](https://www.kaggle.com/datasets/devdope/900k-spotify/data?select=spotify_dataset.csv)

- Import the file `spotify_dataset.csv` into the `data` folder or from our Drive

- Run the script `clean-good4.py` to obtain the cleansed csv


## 🛠️ Python Environment Setup

This project uses a reproducible Conda environment defined in environment.yml.
To create and activate the environment, run the following commands:

#### Install Conda (if needed)

You can use Anaconda or Miniconda:
https://docs.conda.io/en/latest/miniconda.html

#### Create the environment

From the root of the repository:

`conda env create -f environment.yml`

This will install all required dependencies with correct versions.

#### Activate the environment
`conda activate iao`

#### Install additional packages
`conda install -c conda-forge streamlit altair pandas matplotlib`
