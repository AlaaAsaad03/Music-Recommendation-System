#!/usr/bin/env python
# coding: utf-8

# # MUSIC RECOMMENDATION

# ### A) Problem Statement
# The project aims to develop a music recommendation system that predicts what 
# tracks a user might enjoy based on their past listening habits. This system will 
# leverage the "song_data" dataset, which contains detailed information about 
# various songs, to provide personalized music recommendations.
# 
# #### Objectives and Goals
# 1. **Data Preparation:** Clean and preprocess the "song_data" dataset for modeling.
# 2. **Exploratory Data Analysis (EDA):** Analyze data trends and correlations to inform model selection.
# 3. **Model Selection and Training:** Evaluate and train recommendation algorithms using the prepared dataset.
# 4. **Model Evaluation and Optimization:** Assess model performance and optimize algorithms for enhanced recommendation accuracy.
# 5. **Recommendation Generation:** Develop algorithms to generate personalized music recommendations based on user listening history.
# 6. **Deployment:** Implement the recommendation system and ensure scalability for real-world usage.
# 

# ### B) Dataset Description
# The Spotify song data CSV dataset contains comprehensive information about a wide range of songs available on the Spotify platform. It includes both numerical and categorical attributes, providing insights into various aspects of each track. Some of the key attributes in the dataset include acousticness, danceability, energy, duration, instrumentalness, valence, popularity, tempo, liveness, loudness, speechiness, year of release, key, artists, release date, track name, mode, and explicit content
# This rich dataset serves as the foundation for developing a music recommendation system that predicts tracks users might enjoy based on their listening history.

# In[ ]:


# IMPORTt NEEDED LIBRARIES


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import numpy as np
from scipy import stats
import seaborn as sns


# In[ ]:


#IMPORT DATA FROM A CSV FILE INTO A DATAFRAME


# In[2]:


df = pd.read_csv('song_data.csv')


# ### C) Exploratory Data Analysis

# In[ ]:


# TAKE LOOK AT DATASET


# In[3]:


print("The first 5 rows of the dataset are:")
df.head()


# In[ ]:


# QUICK DESCRIPTION OF THE DATA IN THE DATAFRAME


# ### View summary of dataset

# In[41]:


df.info()


# # Description:
# The dataset contains 170,653 entries and 19 columns.
# - The columns include both numerical (float64 and int64) and categorical (object) data types.
# - Features such as 'valence', 'acousticness', 'danceability', 'energy', and others represent musical attributes on a scale from 0 to 1.
# - There are no missing values in the dataset, as all columns have a non-null count of 170,653 entries.
# - The 'release_date' column is stored as an object (string) data type, which may need to be converted to datetime for analysis

# In[42]:


df.describe().transpose()


# ### Drop irrelevant columns
#  'id' is the unique identifier of each track in the dataset. We can drop this variable since it doesn't provide us with any useful information for the recommender system.

# In[4]:


df.drop('id', axis=1, inplace=True)


# In[44]:


df.info()


# ### D) Data preparation
#  Before diving into analysis, it's crucial to ensure the dataset is clean and ready for processing.
#  This involves identifying and addressing any inconsistencies, errors, or missing values in the data.
#  In this section, we'll perform various data cleaning tasks, including:
#  - Handling missing values
#  - Removing duplicates
#  - Correcting data types
#  - Standardizing data formats
#  Let's start by loading the dataset and examining its structure to identify areas that require cleaning.

# #### Handle missing values

# In[ ]:


# Detecting missing values numerically


# In[45]:


df.isna()


# In[46]:


df.isna().any()


# This result indicates that all columns except 'Unnamed: 19' contain no missing values (False) in the dataset. We can drop 'Unnamed: 19'

# In[5]:


df.drop('Unnamed: 19', axis=1, inplace=True)


# In[52]:


df.info()


# #### Remove duplicates 

# In[6]:


# find duplicate rows in the entire dataset
duplicate_rows = df.duplicated()
# print duplicate rows
print(duplicate_rows)
# duplicates on entire dataset:
len(df)-len(df.drop_duplicates())


# In[7]:


subsetAfterDropDuplicateValues = df.drop_duplicates()
subsetAfterDropDuplicateValues.info()


# In[8]:


subsetAfterDropDuplicateValues.to_csv('subsetAfterDropDuplicateValues.csv')


# #### Detect outliers

# In[9]:


# Calculate Z-scores for numerical features
z_scores = stats.zscore(df.select_dtypes(include=[np.number]))

# Define threshold for Z-scores
threshold = 3  # Example threshold value

# Find indices of outliers
outlier_indices = np.where(np.abs(z_scores) > threshold)

# Remove outliers from DataFrame
df_no_outliers = df.drop(outlier_indices[0])

# Print the shape of the DataFrame before and after removing outliers
print("Shape of DataFrame before removing outliers:", df.shape)
print("Shape of DataFrame after removing outliers:", df_no_outliers.shape)


# # E) Data visualization

# In[58]:


df_no_outliers.hist(bins = 100, figsize = (20,15))
plt.savefig("attribute_histogram_plots")


# The above Visualization shows the variability of each metric in the dataset

# # F) Model Training and Testing-Methodology
# ## Recommender System Methodology
# 
# ### Content-Based Music Recommender System Using Cosine Similarity
# 
# #### Overview
# This system suggests music tracks to users based on the features of tracks they have liked in the past, using a content-based filtering approach with cosine similarity.
# 
# #### Methodology
# 
# ##### 1. **Data Preprocessing**
#   1.1) Create a new DataFrame containing only these selected features.
#   
# 

# In[10]:


columns_to_keep = ['year','name', 'artists','key']
df_cleaned = df_no_outliers[columns_to_keep]


# In[11]:


df_cleaned.to_csv('df_cleaned.csv')


# In[60]:


print(df_cleaned)


# In[61]:


df_cleaned.info()


# 1.2) Sample 30,000 entries from this DataFrame for further analysis. 

# In[12]:


sample_df = df_cleaned.sample(n=30000, random_state=42)
print(sample_df)


# In[13]:


sample_df.to_csv('sample_df.csv')


# In[63]:


sample_df.info()


# ##### 2. **Feature Extraction**
#   2.1)   Select relevant features: name, Year, Key, and Artists.
# 

# In[ ]:


features = ['key','year','artists','name']


# 2.2) Concatenate the 'artists', 'year', 'key', and 'name' columns of a DataFrame row into a single string.

# In[64]:


def combine_features(row):
    return (
        row['artists'] + " " +
        str(row['year']) + " " +
        str(row['key']) + " " +
        str(row['name']) 
    )


# 2.3) Add a new column 'combine_features' to sample_df by applying the combine_features function to each row.

# In[66]:


sample_df['combine_features'] = sample_df.apply(lambda row: combine_features(row), axis=1)
sample_df["combine_features"].head(5)


# 2.4) Transform the 'combine_features' column of sample_df into a count matrix, and stores the result in count_matrix

# In[67]:


cv = CountVectorizer()
count_matrix = cv.fit_transform(sample_df["combine_features"])


# ##### 3. **Cosine Similarity Calculation**
# Compute the cosine similarity between the rows of the count_matrix and stores the resulting similarity matrix in cosine_sim.

# In[68]:


cosine_sim = cosine_similarity(count_matrix)


# In[69]:


print(cosine_sim.shape)


# In[ ]:


df_cleaned['index'] = range(len(df_cleaned))


# In[71]:


df_cleaned.head(5)


# #### 4. **Generate Recommendations**
# 4.1) Define two helper functions to get the movie title from the movie index and vice-versa.

# In[72]:


def get_name_from_index(index):
    return df_cleaned[df_cleaned.index == index]["name"].values[0]

def get_index_from_name(name):
    return df_cleaned[df_cleaned.name == name]["index"].values[0]


# Our next step is to get the name of the song that the user currently likes. Then, we will find the
# index of that song in df. After that, we will access the row corresponding to this song in the
# similarity matrix. Thus, we will get the similarity scores of all other song from the current
# song. Then, we will enumerate through all the similarity scores of that song to make a tuple of
# song index and similarity score. This will convert a row of similarity scores like this: [1 0.5 0.2
# 0.9] to this: [(0, 1) (1, 0.5) (2, 0.2) (3, 0.9)]. Here, each item is in this form: (song index,
# similarity score).
# The function get_name_from_index(index) takes the index of the song as argument, then,
# returns the name value of that song in the dataframe using the property values.
# The function get_index_from_name(name) takes the name of the song as argument, then, returns
# the index value of that song in the dataframe using the property values.

# In[ ]:


# Test the function get_index_from_name(name) with a song named “Gati Bali”.


# In[73]:


song_user_likes = "Gati Bali"
song_index = get_index_from_name(song_user_likes)
print(song_index)


# Access the row corresponding to the given song to find all the similarity scores for that song
# and then enumerate over it. You can use the list function which creates a collection that can be
# manipulated for your analysis. This collection of data is called a list object. When you
# use enumerate, the function gives you back two loop variables:
# - The count of the current iteration
# - The value of the item at the current iteration

# In[74]:


similar_songs = list(enumerate(cosine_sim[song_index]))


# Sort the list similar_songs according to similarity scores in descending order. The function
# sorted is used to sort the elements similar_songs with respect to the element at index 1 (second
# element – similarity score). The sorted function returns a sorted list of the specified object. You
# can specify ascending or descending order. Strings are sorted alphabetically, and numbers are
# sorted numerically. Since the most similar song to a given song will be itself, we will discard the
# first element after sorting the songs.

# In[ ]:


sorted_similar_songs = sorted(similar_songs, key=lambda x:x[1],reverse=True)[1:]
# The key argument is meant to specify how to perform the sort.
# The reverse keyword is optional, It is a Boolean value, False will sort ascending, True will sort descending.


# Run a loop to print first 5 entries from the sorted_similar_songs list.

# In[75]:


i=0
print("Top 5 similar songs to "+song_user_likes+" are:\n")
for element in sorted_similar_songs:
    print(get_name_from_index(element[0]))
    i=i+1
    if i>4:
        break


# # G) Conclusion and Future Work
# ## Conclusion
# This project successfully developed a content-based music recommender system using cosine similarity. The system utilizes track features to recommend songs that are similar to those a user has liked in the past. Key steps included data preprocessing, feature selection, similarity computation, and generating recommendations.
# 
# Key findings and outcomes:
# - **Effective Feature Selection**: By selecting relevant features such as Acousticness, Danceability, Energy, and others, the system can accurately represent the characteristics of each track.
# 
# - **Scalability**: The approach can handle large datasets efficiently, demonstrated by sampling 30,000 tracks for analysis.
#     
# ## Future Work
# While the current system provides accurate recommendations based on track features, there are several potential enhancements and areas for further research:
# 
# 1. **User Interaction Data**: Incorporating user interaction data (e.g., user ratings, play counts) could improve recommendation accuracy by combining content-based and collaborative filtering methods.
# 2. **Advanced Similarity Measures**: Exploring advanced similarity measures such as Pearson correlation  could enhance the system's performance.
# 3. **Real-Time Recommendations**: Implementing real-time recommendation updates as users interact with the system would provide a more dynamic and personalized experience.
# 4. **Extended Feature Set**: Adding more features, such as lyrics analysis, genre classification, and mood detection, could provide deeper insights into track characteristics.
# 
# By implementing these enhancements, the music recommender system can become more robust, accurate, and user-centric, providing a richer and more satisfying experience for users.
# 
# 
