
# Music Recommendation System

## Overview

This repository contains the implementation of a music recommendation system using Spotify’s song dataset. The system predicts which tracks a user might enjoy based on their past listening preferences. The system utilizes a content-based recommendation approach, leveraging various audio features such as acousticness, danceability, and energy, among others, to generate personalized music recommendations.

The project uses the "song_data" dataset, which contains 170,653 rows and 19 columns, including numerical and categorical features like track popularity, artists, tempo, and more.

## Features

- **Content-Based Filtering:** The recommendation system is based on the content of the songs a user has listened to, comparing various audio features.
- **Cosine Similarity:** The system computes the similarity between songs using cosine similarity, providing a ranked list of recommended songs based on the user's listening history.
- **Exploratory Data Analysis (EDA):** Initial analysis to understand the dataset's structure, trends, and correlations.
- **Data Preprocessing:** Cleaning the data by removing duplicates, handling missing values, and outlier detection.
- **Modeling and Evaluation:** Creating a model that generates recommendations based on the features of songs the user has liked in the past.

## Dataset

The dataset used in this project, `song_data.csv`, contains 19 columns:

### Numerical Columns:
1. **Acousticness:** Relative metric of the track being acoustic (0 to 1).
2. **Danceability:** How suitable the track is for dancing (0 to 1).
3. **Energy:** The energy level of the track (0 to 1).
4. **Duration_ms:** Length of the track in milliseconds.
5. **Instrumentalness:** The ratio of instrumental content in the track (0 to 1).
6. **Valence:** The positiveness of the track (0 to 1).
7. **Popularity:** Popularity score of the track (0 to 100).
8. **Tempo:** The tempo of the track (BPM).
9. **Liveness:** The likelihood that the track was recorded live (0 to 1).
10. **Loudness:** The relative loudness of the track (in dB).
11. **Speechiness:** Measure of human voice content in the track (0 to 1).
12. **Year:** Year of release.
13. **Id:** Unique identifier for the track.

### Categorical Columns:
1. **Key:** Musical key of the track (0 to 11).
2. **Artists:** List of artists credited with the track.
3. **Release_date:** Date the track was released.
4. **Name:** Name of the track.
5. **Mode:** Whether the track has a major or minor chord progression (binary).
6. **Explicit:** Whether the track contains explicit content (binary).

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/music-recommendation-system.git
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Data Preprocessing

Run the script to load and clean the dataset:

```bash
python data_preprocessing.py
```

This script performs the following:
- Loads the `song_data.csv` dataset.
- Drops irrelevant columns and handles missing values.
- Removes duplicates and outliers.

### Step 2: Exploratory Data Analysis (EDA)

You can analyze the dataset for trends and correlations with:

```bash
python eda.py
```

This will display visualizations like histograms and correlation heatmaps.

### Step 3: Model Training

Train the recommendation model using the content-based filtering approach:

```bash
python train_model.py
```

This will:
- Extract relevant features such as `artists`, `year`, `key`, and `name`.
- Create a count vectorizer for text-based features.
- Compute cosine similarity between the songs.

### Step 4: Generate Recommendations

After training, you can generate recommendations for a song using the following:

```bash
python recommend.py --song "Song Name"
```

Replace `"Song Name"` with the track title you wish to get recommendations for.

## Directory Structure

```
.
├── data_preprocessing.py       # Data cleaning and preprocessing script
├── eda.py                      # Exploratory Data Analysis script
├── recommend.py                # Song recommendation script
├── train_model.py              # Model training script
├── requirements.txt            # Python dependencies
├── song_data.csv               # Dataset used for the recommendation system
└── README.md                   # This file
```

## Requirements

- Python 3.x
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- SciPy
- Missingno

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Contributing

Feel free to fork the repository and submit pull requests for any improvements or bug fixes. Make sure to follow the standard code practices and write tests where necessary.

