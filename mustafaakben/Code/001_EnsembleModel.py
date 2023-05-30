#########################################################################
#########################################################################
### ANCHOR Import Libraries

import os
import gc
import re
import sys
import h2o
import time
import glob
import nltk
import spacy
import FRUFS
import string
import pathlib
import warnings
import twinning
import wordninja
import subprocess
import numpy as np
import pandas as pd
import networkx as nx
import textstat as ts
import tensorflow_text
import tensorflow as tf
import tensorflow_hub as hub
import textdescriptives as td


from numba import njit
from scipy.linalg import eigh
from h2o.automl import H2OAutoML
from xgboost import XGBRegressor
from nltk.corpus import stopwords
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import pdist
from InstructorEmbedding import INSTRUCTOR
from sklearn.cluster import BisectingKMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sentence_transformers import SentenceTransformer
from sklearn.feature_selection import VarianceThreshold

nltk.download('stopwords')



# Check the whether the GPU is available
def is_gpu_available():
    try:
        # If this command fails, there's no GPU available.
        # This command lists all NVIDIA GPUs that are visible to the driver.
        result = subprocess.run(["nvidia-smi", "--list-gpus"], check=True, stdout=subprocess.PIPE)
        print(os.system("nvidia-smi"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        # subprocess.CalledProcessError means the command reported an error
        # FileNotFoundError means the command isn't installed
        return False

    # If there was any output, then at least one GPU is available
    return len(result.stdout) > 0


is_gpu_available()
np.set_printoptions(precision=3)


#########################################################################
#########################################################################
### ANCHOR Data Load

# Flag to control the target rating for temporary use
TARGET_RATING = 6

# Flags to control column handling
multi_columns = False
multi_columns_by_name = True


# Define the path to the CSV file
csv_file_path = pathlib.Path("data/FullData.csv") 

# Load the data using pandas. The CSV file is located at the defined path.
data = pd.read_csv(csv_file_path)

# Uncomment the line below if you want to shuffle your data and reset the index
# data = data.sample(frac=1).reset_index(drop=True)

# Print the columns of the DataFrame. Useful to understand the structure of your data.
print(data.columns)



#########################################################################
#########################################################################
### ANCHOR Some Utility Functions


def find_groups(corr_matrix, threshold):
    """
    Function to find cluster|groups of highly correlated features.
    """
    i, j = np.where(corr_matrix >= threshold)
    mask = i < j
    edges = zip(corr_matrix.index[i[mask]], corr_matrix.columns[j[mask]])
    
    # Create graph from edges and extract connected components
    G = nx.Graph()
    G.add_edges_from(edges)
    return list(nx.connected_components(G))


def pca_with_network_cluster(X, n_iter=5, threshold=0.8):
    """
    Function to perform PCA by group for each iteration on the input data.
    """
    # Standardization
    df = pd.DataFrame(StandardScaler().fit_transform(X))
    df.columns = [f'col{i}' for i in range(len(df.columns))]
    
    # Iteratively group highly correlated features and replace them with the first PCA component
    for each_iter in range(n_iter):
        print(f"ITERATION :{each_iter}")
        corr_matrix = df.corr().abs()
        
        # Group highly correlated features
        groups = find_groups(corr_matrix, threshold)
        
        # Perform PCA for each group and replace original features
        for i, group in enumerate(groups):
            group = list(group)

            # PCA
            pca = PCA(n_components=1)
            reduced_features = pca.fit_transform(df[group])
            
            # Replace original features with PCA components
            new_column = f'PCA_group_{i}'
            df[new_column] = reduced_features
            df.drop(group, axis=1, inplace=True)
    
    return df.values


# Function to compute correlation coefficient for presenation purposes
def compute_corrcoef(X, Y, idx, which_column,top_n = 20):
    r = np.corrcoef(Y[idx, which_column], X[idx], rowvar=False)[0]
    r[np.isnan(r)] = 0
    r = np.abs(np.round(r, 3))
    # Sort and show top_n correlations
    return np.flip(np.sort(r))[1:top_n]


def clc():
    gc.collect()
    os.system("clear")


## Fast Correlation Coefficient

@njit
def pearson_correlation(x, y):
    n = x.shape[0]
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x * x)
    sum_y2 = np.sum(y * y)
    sum_xy = np.sum(x * y)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))

    if denominator == 0:
        return 0

    return numerator / denominator



def drop_min_correlation(X,target,test_size=400, n_cros_val=100,percentage_of_features_to_drop=.25, verbose=False):
    droping_n = int(X.shape[1]*percentage_of_features_to_drop)

    WORST_COLS = []
    for i in range(n_cros_val):
        if verbose:
            if i % 20 == 0:
                print("BootStrapping: ",i)
        
        cols = []
        cors = []
        
        for i in range(X.shape[1]):
            sample_idx = np.random.choice(len(X),test_size)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # r = np.corrcoef(X[sample_idx,i],target[sample_idx])[0,1]
                r = pearson_correlation(X[sample_idx,i],target[sample_idx])
            r = np.abs(r)
            if np.isnan(r):
                r = 0
            cols.append(i)
            cors.append(r)
            # print(f"{r:.2f}")
        temp_df = pd.DataFrame({'cols':cols,'cors':cors})
        worst_cols = list(temp_df.sort_values('cors')[:droping_n].index)
        WORST_COLS.extend(worst_cols)  
            
    temp_df = pd.DataFrame(WORST_COLS)
    temp_df = temp_df.value_counts()
    idx = [i[0] for i in temp_df.index]
    OMIT_THESE_COLS = idx[:droping_n]

    COLS = np.setdiff1d(range(X.shape[1]),OMIT_THESE_COLS)
    return COLS


# Function to check the cross validation score for the random forest model
def cross_val_forest_score(X, Y, test_train_idx, target_rating, cv=5, n_estimators=100, max_depth=4, max_samples=.8, random_state=10):
    temp_X = X[test_train_idx]
    random_forest_regressor = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples, n_jobs=-1, random_state=random_state)
    r2_scores = cross_val_score(random_forest_regressor, temp_X,
                                Y[test_train_idx, target_rating], cv=cv, scoring='r2')
    return print("Cross validation result:", f"{np.mean(r2_scores)**(1/2):.3f}")



clc()


#########################################################################
#########################################################################
### ANCHOR Data Preprocessing

# Create boolean masks for rows marked as "TRAIN" in the "test_train" column
test_train_idx = data.test_train == "TRAIN"
test_train_idx = test_train_idx.values

# Create boolean masks for rows with "additional_data" equals 
# This will be the final test data for the competition

test_indices = data.additional_data == 1
test_indices = test_indices.values


# Get the "TYPE2" column values which shows the types of simulation games
type_values = data.TYPE2.values


def clean_whitespace(text):
    """
    Cleans a text string by splitting it into words, when it is hard to detect whitespaces.
    
    Args:
    text: str, the string to be cleaned.
    
    Returns:
    str, the cleaned string. If input is NaN, returns NaN.
    """
    if pd.isna(text):  # If text is NaN, return NaN
        return np.nan
    else:  # Else split the string into words and join with a space
        return " ".join(wordninja.split(text))

# Example use
print(clean_whitespace("Thisisateststring"))


# Apply the clean_whitespace function to all text columns from 'text_exercise_4' onwards
data.loc[:, 'text_exercise_4':] = data.loc[:, 'text_exercise_4':].applymap(clean_whitespace)



#########################################################################
#########################################################################
### ANCHOR Combines different column pairs by adding leading questions to text


# Function to add leading question and combine two columns
def add_question_and_combine_columns(col1, col2, question):
    """
    Adds a leading question to each element of two columns and combines them.
    If an element is NaN, it remains NaN in the combined result.

    Args:
    col1, col2: array-like, two columns of data to be combined.
    question: str, the leading question to be added.

    Returns:
    list, the combined column data.
    """
    col1 = [f"{question}\n{element}" if not pd.isna(element) else np.nan for element in col1]
    combined = []
    for t1, t2 in zip(col1, col2):
        if pd.isna(t1) and pd.isna(t2):
            combined.append(np.nan)
        elif pd.isna(t1) and not pd.isna(t2):
            combined.append(f"{question}\n{t2}")
        elif not pd.isna(t1) and pd.isna(t2):
            combined.append(t1)
        else:
            combined.append(t1 + "\n" + t2)
    return combined


# Add leading questions to and combine pairs of columns
type1_1_combined = add_question_and_combine_columns(data['TYPE_1_TWO'].values, data['TYPE_1_THREE'].values, 
                                                    "What are the major categories of issues or problems? Why?")
type1_2_combined = add_question_and_combine_columns(data['TYPE_1_FOUR'].values, data['TYPE_1_FIVE'].values,
                                                    "What were the three most important and the three least important e-mails? Why?")
type1_3_combined = add_question_and_combine_columns(data['TYPE_1_ONE'].values, data['TYPE_1_SIX'].values,
                                                    "How did you handle each challenge, in an order or a different way?")
type2_1_combined = add_question_and_combine_columns(data['TYPE_2_ONE'].values, data['TYPE_2_TWO'].values,
                                                    "What are the major categories of issues or problems? Why?")
type2_2_combined = add_question_and_combine_columns(data['TYPE_2_THREE'].values, data['TYPE_2_FOUR'].values,
                                                    "What are the relationship among the messages you received in your inbox?")

# Combine TYPE_1_OTHER and TYPE_2_OTHER into one column, omitting NaN inputs
type_others_combined = data['TYPE_1_OTHERS'].fillna("") + " " + data['TYPE_2_OTHER'].fillna("")

# Remove original columns and add new combined ones
data = data.loc[:, ~data.columns.str.contains("TYPE_")]
data = data.assign(TYPE_1_1_COMB=type1_1_combined,
                   TYPE_1_2_COMB=type1_2_combined,
                   TYPE_1_3_COMB=type1_3_combined,
                   TYPE_2_1_COMB=type2_1_combined,
                   TYPE_2_2_COMB=type2_2_combined,
                   TYPE_OTHERS=type_others_combined)


## Sample a nan value from the text to see what it made
print(data.loc[~data['TYPE_1_1_COMB'].isna(), 'TYPE_1_1_COMB'].sample(1).values[0])




#########################################################################
#########################################################################
### ANCHOR Performs data imputation, fills missing text data 
#          with an empty string, and does some text cleaning.

# Function to replace extra whitespaces in the DataFrame
def replace_extra_whitespaces(df):
    """
    Replaces occurrences of two or more consecutive spaces with a single space in a DataFrame.

    Args:
    df: pandas DataFrame

    Returns:
    pandas DataFrame with replaced extra whitespaces.
    """
    return df.replace(r'\s+', ' ', regex=True)


# Select the text columns from 'text_exercise_4' onward
X_text = data.loc[:, "text_exercise_4":]

# Select the rating columns
rating_columns = data.columns[data.columns.str.contains("rating")].to_list()

Y = data.loc[:, rating_columns]

# Select the portion of the Y DataFrame to impute (where test_train_idx is True)
to_impute = Y[test_train_idx]

# Initialize a KNN imputer
imputer = KNNImputer(n_neighbors=3, weights="uniform")

# Apply imputation on the selected part of Y DataFrame and round the values to the nearest integer
Y.loc[test_train_idx, :] = np.round(imputer.fit_transform(to_impute), 0)

# Verify that there are no more missing values in the imputed portion of the Y DataFrame
assert Y.loc[test_train_idx].isna().sum().sum() == 0, "There are still missing values after imputation!"

# Fill missing text data with an empty string
X_text = X_text.fillna("")

# Replace extra whitespaces such as two or more spaces in X_text DataFrame
X_text = replace_extra_whitespaces(X_text)

# Replace all occurrences of "J J" with "JJ" in X_text DataFrame
X_text = X_text.replace("J J", "JJ", regex=True)
X_text = X_text.replace("J J", "JJ", regex=True)

## Search for "J J" in X_text for each column
print(X_text.apply(lambda x: x.str.contains("J J")).sum())


data[X_text.columns] = X_text
data[Y.columns] = Y





#########################################################################
#########################################################################
### ANCHOR Conditionally transforms the data, renames columns, 
#         selects relevant columns, and imputes missing values.

# Define a function for renaming the columns based on types
def rename_columns(temp_df, each_type):
    """
    Rename columns of DataFrame based on each type.

    Args:
    temp_df: pandas DataFrame
    each_type: unique TYPE2 value

    Returns:
    pandas DataFrame with renamed columns.
    """
    text_filter = temp_df.columns.str.contains('text_') | temp_df.columns.str.contains('TYPE_')
    text_names = [col for col in temp_df.columns[text_filter] if col != 'text_exercise_final']

    renamed = {each_col: f"t_{each_type}_" + each_col for each_col in text_names}
    temp_df.rename(columns=renamed, inplace=True)

    return temp_df, [f"t_{each_type}_" + each_col for each_col in text_names]


## If multi_columns is True, perform the following operations
## With this, the dataset will be masked coded based on the simulation type

if multi_columns:
    split_data = []
    types = data.TYPE2.unique()

    for each_type in types:
        type_filter = data.TYPE2 == each_type
        temp_df = data[type_filter]

        # Rename columns
        temp_df, text_names = rename_columns(temp_df, each_type)

        # Calculate the text length for each text column
        # If it is less than 2, assign as np.nan
        for each_col in text_names:
            temp_df.loc[:, each_col] = temp_df.loc[:, each_col].apply(lambda x: np.nan if len(x) < 2 else x)
            
        # Select only those columns where more than 60% of the data is non-missing
        select = temp_df[text_names].isna().sum() / len(temp_df) < .6
        select = select.index[select]
        temp_right = temp_df[select]
        temp_left = temp_df.loc[:, "test_train":"rating_decision_making_final_score"]

        temp_df = pd.concat([temp_left, temp_right], axis=1)
        split_data.append(temp_df)

    # Perform a full outer join on the list of DataFrames
    data = pd.concat(split_data, axis=0, join='outer')

    test_train_idx = data.test_train == "TRAIN"
    test_train_idx = test_train_idx.values
    TEST_IDX = data.additional_data == 1 
    X_TYPE = data.TYPE2.values
    
    # Get text names
    text_filter = data.columns.str.contains('text_') | data.columns.str.contains('TYPE_')
    text_names = [col for col in data.columns[text_filter] if col != 'text_exercise_final']

    X_text = data.loc[:, text_names]
    Y = data.loc[:, "rating_chooses_appropriate_action":"rating_decision_making_final_score"]
    to_impute = Y[test_train_idx]

    # Initialize a KNN imputer
    imputer = KNNImputer(n_neighbors=3, weights="uniform")
    # Apply imputation on the selected part of Y DataFrame and round the values to the nearest integer
    Y.loc[test_train_idx, :] = np.round(imputer.fit_transform(to_impute), 0)

    # Verify that there are no more missing values in the imputed portion of the Y DataFrame
    assert Y.loc[test_train_idx].isna().sum().sum() == 0, "There are still missing values after imputation!"

    # Fill missing text data with an empty string
    X_text = X_text.fillna("")



## If multi_columns_by_name is True, perform the following operations
## With this, the dataset will be masked coded based on the simulation type

if multi_columns_by_name:
    split_data = []
    types = data.playerName.unique()

    for each_type in types:
        type_filter = data.playerName == each_type
        temp_df = data[type_filter]

        # Rename columns
        # Import warnings library
        
        # Open a warning scope and ignore the warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            temp_df, text_names = rename_columns(temp_df, each_type)
        
        # Calculate the text length for each text column
        # If it is less than 2, assign as np.nan
        for each_col in text_names:
            temp_df.loc[:, each_col] = temp_df.loc[:, each_col].apply(lambda x: np.nan if len(x) < 2 else x)
            

        # Select only those columns where more than 60% of the data is non-missing
        select = temp_df[text_names].isna().sum() / len(temp_df) < .6
        select = select.index[select]
        temp_right = temp_df[select]
        temp_left = temp_df.loc[:, "test_train":"rating_decision_making_final_score"]

        temp_df = pd.concat([temp_left, temp_right], axis=1)
        split_data.append(temp_df)

    # Perform a full outer join on the list of DataFrames
    data = pd.concat(split_data, axis=0, join='outer')

    test_train_idx = data.test_train == "TRAIN"
    test_train_idx = test_train_idx.values
    TEST_IDX = data.additional_data == 1 
    X_TYPE = data.TYPE2.values
    
    # Get text names
    text_filter = data.columns.str.contains('text_') | data.columns.str.contains('TYPE_')
    text_names = [col for col in data.columns[text_filter] if col != 'text_exercise_final']

    X_text = data.loc[:, text_names]
    Y = data.loc[:, "rating_chooses_appropriate_action":"rating_decision_making_final_score"]
    to_impute = Y[test_train_idx]

    # Initialize a KNN imputer
    imputer = KNNImputer(n_neighbors=3, weights="uniform")
    # Apply imputation on the selected part of Y DataFrame and round the values to the nearest integer
    Y.loc[test_train_idx, :] = np.round(imputer.fit_transform(to_impute), 0)

    # Verify that there are no more missing values in the imputed portion of the Y DataFrame
    assert Y.loc[test_train_idx].isna().sum().sum() == 0, "There are still missing values after imputation!"

    # Fill missing text data with an empty string
    X_text = X_text.fillna("")
    
## Print the shape of the data and the column names of the X_text and Y DataFrames
print(data.shape)
print(X_text.columns)
print(Y.columns)




#########################################################################
#########################################################################
### ANCHOR Define Target Rating from the Y Dataframe to Numpy

Y_all = Y.values


#########################################################################
#########################################################################
### ANCHOR Semantic Embedding model



def get_embed_model(model_architecture, model_name):
    """
    Returns an embedding model based on the specified architecture and model name.
    
    Parameters:
    model_architecture (str): The architecture of the model. Options are "semantic_embedding", "sentence_embedding", "instruct_model".
    model_name (str): The specific model to use. Options for each architecture are as follows:
        "semantic_embedding": ["universal-sentence-encoder-large", "universal-sentence-encoder", "t5-base", "t5-large", "t5-xl"]
        "sentence_embedding": ["all-mpnet-base-v2", "all-roberta-large-v1", "stsb-roberta-large", "multi-qa-mpnet-base-dot-v1"]
        "instruct_model": ["instructor-large", "instructor-xl"]

    Returns:
    embed_model: The desired embedding model.
    """
    # Define a dictionary of models for each architecture
    models = {
        "semantic_embedding": {
            "universal-sentence-encoder-large": "https://tfhub.dev/google/universal-sentence-encoder-large/5",
            "universal-sentence-encoder": "https://tfhub.dev/google/universal-sentence-encoder/4",
            "t5-base": "https://tfhub.dev/google/sentence-t5/st5-base/1",
            "t5-large": "https://tfhub.dev/google/sentence-t5/st5-large/1",
            "t5-xl": "https://tfhub.dev/google/sentence-t5/st5-3b/1"
        },
        "sentence_embedding": {
            "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
            "all-roberta-large-v1": "sentence-transformers/all-roberta-large-v1",
            "stsb-roberta-large": "stsb-roberta-large",
            "multi-qa-mpnet-base-dot-v1": "sentence-transformers/multi-qa-mpnet-base-dot-v1"
        },
        "instruct_model": {
            "instructor-large": "hkunlp/instructor-large",
            "instructor-xl": "hkunlp/instructor-xl"
        }
    }

    # Check if the architecture and model are valid
    if model_architecture in models and model_name in models[model_architecture]:
        model_url = models[model_architecture][model_name]

        if model_architecture == "semantic_embedding":
            embedding_model = hub.KerasLayer(model_url)
            ins = tf.keras.layers.Input(shape=(), dtype=tf.string)
            net = embedding_model(ins)
            embed_model = tf.keras.Model(ins, net)

        elif model_architecture == "sentence_embedding":
            embed_model = SentenceTransformer(model_url)

        elif model_architecture == "instruct_model":
            embed_model = INSTRUCTOR(model_url)

    else:
        raise ValueError("Invalid model architecture or model name.")

    return embed_model



# Define the model architecture and model name
MODEL_ARCHITECTURE = "semantic_embedding"
model_name = "t5-large"
embed_model = get_embed_model(MODEL_ARCHITECTURE, model_name)

# Example use
test_sentence = "This is a test sentence."


if MODEL_ARCHITECTURE == "semantic_embedding":
    result = embed_model.predict([test_sentence])
elif MODEL_ARCHITECTURE == "sentence_embedding":
    result = embed_model.encode([test_sentence])
elif MODEL_ARCHITECTURE == "instruct_model":
    instruction = 'Represent the workplace conversation for classifying the conversation:'
    result = embed_model.encode([[instruction, test_sentence]])
else:
    raise ValueError(f"Invalid MODEL_ARCHITECTURE: {MODEL_ARCHITECTURE}")

print(result[0][:50])



#########################################################################
#########################################################################
# ANCHOR Semantic Embedding by Columns


# Download the stopwords and word_tokenize from nltk
# nltk.download('stopwords')

cachedStopWords = stopwords.words("english")


def preprocess_text(text, 
                    lower_case=False,
                    remove_stop_words=False,
                    remove_punctuation=False,
                    clear_double_spaces=False,
                    change_spelling=False):
    """
    Function to preprocess the input text by performing following operations:
    - Lower casing
    - Stop word removal
    - Punctuation removal
    - Clear double spaces
    - Spelling changes
    """
    if lower_case:
        text = text.lower()

    if remove_stop_words:
        text = word_tokenize(text)
        text = ' '.join([word for word in text if word not in cachedStopWords])

    if remove_punctuation:
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)

    if clear_double_spaces:
        text = re.sub(r"(\s{2,})", " ", text)

    if change_spelling:
        text = re.sub(r'\bj j \b', 'jj ', text)
        text = re.sub(r'\b j j\b', ' jj ', text)
        text = re.sub(r'\b j j \b', ' jj ', text)
        text = re.sub(r'\b paxton\b', ' jj ', text)
        text = re.sub(r'\bpaxton \b', ' jj ', text)
        text = re.sub(r'\b paxton \b', ' jj ', text)
        
    return text



def encode_text(X_text, embed_model, architecture, empty_cell_text, print_corr=False):
    """
    Function to encode the text data with a given model and architecture.
    """
    encoded_text = []
    num_cols = X_text.shape[1]
    # col_index = 0
    for col_index in range(num_cols + 1):
        print("COLUMN ENCODING : ", col_index)
        if col_index == num_cols:
            col_index = range(num_cols)
            text = X_text.iloc[:, col_index].fillna("").apply(lambda x: " ".join(x.astype(str)), axis=1)
        else:
            text = X_text.iloc[:, col_index].fillna("")
         
        # Preprocess the text data
        text = text.apply(lambda x: preprocess_text(x,
                                                    lower_case=True,
                                                    remove_stop_words=False,
                                                    remove_punctuation=True,
                                                    clear_double_spaces=True,
                                                    change_spelling=True))

        # Empty cell imputation
        text[text.apply(lambda x: len(x)) < 3] = empty_cell_text
        
        if architecture == "sentence_embedding":
            encoded = embed_model.encode(text.values, batch_size=20, show_progress_bar=True)
        elif architecture == "instruct_model":
            prop_data = [[
                "Represent the workplace conversation for classifying the conversation:",
                t] for t in text.values]
            encoded = embed_model.encode(prop_data, batch_size=10, show_progress_bar=True)
        elif architecture == "semantic_embedding":
            encoded = embed_model.predict(text.values, batch_size=10)
        
        if print_corr:
            print("Correlation Coefficient between the encoded text and the target rating")
            print(compute_corrcoef(encoded, Y_all, test_train_idx, TARGET_RATING, top_n=50))
            print("")
        
        encoded_text.append(encoded)
    
    return encoded_text



EMPTY_CELL_TEXT = "There is no inputs!"
ENCODED_TEXT = encode_text(X_text, embed_model, MODEL_ARCHITECTURE, EMPTY_CELL_TEXT, print_corr=True)


X_encoded = np.array(ENCODED_TEXT).swapaxes(0, 1)


cross_val_forest_score(X_encoded.mean(axis=1),
                       Y_all,
                       test_train_idx,
                       TARGET_RATING,
                       cv=5,
                       n_estimators=100,
                       max_depth=4,
                       max_samples=.8,
                       random_state=10)


#########################################################################
#########################################################################
### ANCHOR NLP Features and other statistics with Scapy

## This part might take some time.
## NLP features from Scapy might slightly increase the performance of the model
## "False" is recommended for faster processing


NLP_FEATURE_STATISTICS = False

if NLP_FEATURE_STATISTICS:
    # Small model
    # nlp = spacy.load("en_core_web_sm")
    nlp = spacy.load('en_core_web_lg') 
    nlp.add_pipe("textdescriptives/all")


    # Get number of columns in the dataset
    _, num_columns = X_text.shape

    # Placeholder to store the results
    result = []

    # Loop through each column to extract text descriptions
    for i in range(num_columns + 1):
        print("Processing Column #:", i)
        
        # When reached to last iteration, join all the text from the columns
        if i == num_columns:
            text = X_text.apply(lambda x: " ".join(x.astype(str)), axis=1).values
        else:
            text = X_text.iloc[:, i].values
        
        # Replace empty text with "No inputs"
        text[text == ""] = "No inputs"

        # Extract the text description features
        docs = nlp.pipe(text.tolist())
        features_df = td.extract_df(docs, include_text=False)
        features_df.fillna(0, inplace=True)
            
        result.append(features_df)


    # Handle columns that might not appear in each DataFrame
    column_names = set(col for df in result for col in df.columns)

    # The data is ragged, so we need to ensure all DataFrames have the same columns
    for df in result:
        for column in column_names:
            if column not in df.columns:
                df[column] = 0.0


    column_names = list(column_names)
    # Reorder columns
    result = [each_df[column_names] for each_df in result]


    # Convert result into a numpy array and transpose it
    result_array = np.array(result).astype('float32').swapaxes(0, 1)
    nlp_mean_features = result_array.mean(axis=1)


    # Calculate correlations
    compute_corrcoef(nlp_mean_features, Y_all, test_train_idx, TARGET_RATING, top_n=50)
    # Reshape the array
    result_array = result_array.reshape(result_array.shape[0], -1)
    # Print the new shape
    print(result_array.shape)
    compute_corrcoef(result_array, Y_all, test_train_idx, TARGET_RATING, top_n=50)

    # Preprocessing for further analysis
    # X_STAT = np.hstack([result_array[:, num_columns, :], result_array.mean(axis=1)])
    X_STAT = np.hstack([result_array, nlp_mean_features])
    # Replace inf and nan values with 0
    X_STAT[np.isinf(X_STAT)] = 0
    X_STAT[np.isnan(X_STAT)] = 0

    # Remove features with zero variance
    X_STAT = VarianceThreshold().fit_transform(X_STAT)

    # Apply Network PCA to reduce dimensionality
    X_STAT = pca_with_network_cluster(X_STAT, 3, threshold=.95)
    compute_corrcoef(X_STAT, Y_all, test_train_idx, TARGET_RATING, top_n=50)


    cross_val_forest_score(X_STAT,
                        Y_all,
                        test_train_idx,
                        TARGET_RATING,
                        cv=5,
                        n_estimators=100,
                        max_depth=4,
                        max_samples=.8,
                        random_state=10)




#########################################################################
#########################################################################
# ANCHOR Text Structure and Complexity Information

# Function to handle missing and special value text
def handle_text_values(text_df):
    text_df[text_df == "|"] = np.nan
    text_df[text_df == "EMPTYCELL EMPTYCELL"] = np.nan
    text_df[text_df == "No Inputs"] = np.nan
    return text_df

# Function to get text lengths
def get_text_lengths(text_df):
    return text_df.applymap(lambda x: len(x) if not pd.isna(x) else 0)

# Function to scale and remove zero variance features
def scale_remove_variance(data):
    transform = StandardScaler()
    zero_var = VarianceThreshold()
    data = transform.fit_transform(data)
    data = zero_var.fit_transform(data)
    return data


# Check for NaN in target variable (Y_all)
print(f'Number of NaN values in Y: {np.isnan(Y_all[test_train_idx]).sum()}')


# Handle text values
temp_text = X_text.copy()
temp_text = handle_text_values(temp_text)

# Get text lengths
X_LEN = get_text_lengths(temp_text)
X_LEN['SUM'] = X_LEN.sum(axis=1)

# Scale and remove zero variance features
X_LEN = scale_remove_variance(X_LEN)

# Compute correlation coefficient
corr_coef = compute_corrcoef(X_LEN, Y_all, test_train_idx, TARGET_RATING, top_n=50)
print(f'Correlation Coefficient: \n\n{corr_coef}')


cross_val_forest_score(X_LEN,
                       Y_all,
                       test_train_idx,
                       TARGET_RATING,
                       cv=5,
                       n_estimators=100,
                       max_depth=4,
                       max_samples=.8,
                       random_state=10)

#########################################################################
#########################################################################
### ANCHOR Extract the text complexity features with textstat


# Prepare text for textstat complexity analysis
temp_text = X_text.copy()
temp_text[temp_text == "|"] = np.nan
temp_text[temp_text == "EMPTYCELL EMPTYCELL"] = np.nan
temp_text = pd.DataFrame(temp_text)
temp_text["text_all"] = temp_text.fillna(
    " | ").apply(lambda x: " | ".join(x), axis=1)



function_list = [
    ts.flesch_reading_ease,
    ts.flesch_kincaid_grade,
    ts.flesch_reading_ease,
    ts.flesch_kincaid_grade,
    ts.smog_index,
    ts.coleman_liau_index,
    ts.dale_chall_readability_score,
    ts.difficult_words,
    ts.linsear_write_formula,
    ts.gunning_fog,
    ts.fernandez_huerta,
    ts.szigriszt_pazos,
    ts.gutierrez_polini,
    ts.crawford,
    ts.gulpease_index,
    ts.osman,
    ts.mcalpine_eflaw,
    ts.reading_time,
    ts.syllable_count,
    ts.lexicon_count,
    ts.sentence_count,
    ts.char_count,
    ts.polysyllabcount,
    ts.monosyllabcount
]


result_list = []

# Apply textstat functions to text
for func in function_list:
    try:
        print("Function Name: ", func.__name__)
        scores = temp_text.applymap(lambda x: func(x) if not pd.isna(x) else 0)
        scores["AVE"] = scores.mean(axis=1)
        result_list.append(scores)

        print(compute_corrcoef(scores, Y_all,
              test_train_idx, TARGET_RATING, top_n=50))

    except:
        print(f"Skipping function {func.__name__}")



# Collate all results
result_array = np.array([df.values for df in result_list]).swapaxes(0, 1)

# Process text complexity features
zero_var = VarianceThreshold()
scaler = StandardScaler()
lingo_means = np.mean(result_array, axis=1)
X_LINGO = pd.concat(result_list, axis=1)
X_LINGO = np.hstack([X_LEN, X_LINGO, lingo_means])
X_LINGO = zero_var.fit_transform(X_LINGO)
X_LINGO = scaler.fit_transform(X_LINGO)

# Perform PCA to reduce dimensionality with network clustering
X_LINGO.shape
X_LINGO = pca_with_network_cluster(X_LINGO, 3, threshold=.98)
X_LINGO.shape

# Compute correlation coefficient
compute_corrcoef(X_LINGO, Y_all, test_train_idx, TARGET_RATING, top_n=150)


cross_val_forest_score(X_LINGO,
                       Y_all,
                       test_train_idx,
                       TARGET_RATING,
                       cv=5,
                       n_estimators=100,
                       max_depth=4,
                       max_samples=.8,
                       random_state=10)

#########################################################################
#########################################################################
### ANCHOR Define Diffusion map function


def diffusion_map(X, n_components=30, t=1, epsilon=None, return_eigenvectors=True):
    # Compute the pairwise distances between data points
    distances = squareform(pdist(X, metric='cosine'))
    
    # Estimate the epsilon parameter if not provided
    if epsilon is None:
        epsilon = np.median(distances) / 2
        # epsilon = np.mean(distances) / 2
        if epsilon <= 0.0005:
            epsilon = 1

    # Compute the kernel matrix
    kernel_matrix = np.exp(-distances ** 2 / (2 * epsilon ** 2))

    # Compute the row sums and the diagonal matrix D
    row_sums = kernel_matrix.sum(axis=1)
    D_inv_sqrt = np.diag(1 / np.sqrt(row_sums))

    # Compute the normalized kernel matrix
    K_norm = np.dot(D_inv_sqrt, np.dot(kernel_matrix, D_inv_sqrt))

    # Compute the eigenvalues and eigenvectors of the normalized kernel matrix
    eigvals, eigvecs = eigh(K_norm)

    # Sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    # Compute the diffusion map embedding
    if return_eigenvectors:
        diffusion_map_embedding = eigvecs[:, :n_components] * (eigvals[:n_components] ** t)
    else:
        diffusion_map_embedding = eigvecs[:, :n_components]

    return diffusion_map_embedding


#########################################################################
#########################################################################
### ANCHOR Reduce Semantic Column Embeddings with FeatureAgglomeration


embedding_features = []  # List to store reduced embeddings for each column
N_cluster = 10  # Number of clusters for feature agglomeration

for each in range(X_encoded.shape[1]):
    temp_x = X_encoded[:, each, :]  # Extract the current column
    
    
    # Calculate correlation coefficients between the target variable and the column
    print("Inital Correlation Coefficient:")
    compute_corrcoef(temp_x, Y_all, test_train_idx, TARGET_RATING, top_n=15)
    
    # Apply feature agglomeration to reduce the dimensionality of the column
    featureCombine = FeatureAgglomeration(n_clusters=N_cluster)
    temp_x = featureCombine.fit_transform(temp_x)
    # temp_x = diffusion_map(temp_x, n_components=N_cluster, t=3, epsilon=None, return_eigenvectors=True)

    # Calculate correlation coefficients after feature agglomeration
    print("Reduced Dimension Correlation Coefficient:")
    compute_corrcoef(temp_x, Y_all, test_train_idx, TARGET_RATING, top_n=15)
    print("\n\n")
    
    to_out = temp_x  # Store the reduced embeddings
    embedding_features.append(to_out)

X_col_embeds = np.hstack(embedding_features)  # Combine the reduced embeddings
X_col_embeds.shape



cross_val_forest_score(X_col_embeds,
                       Y_all,
                       test_train_idx,
                       TARGET_RATING,
                       cv=5,
                       n_estimators=100,
                       max_depth=4,
                       max_samples=.8,
                       random_state=10)



#########################################################################
#########################################################################
### ANCHOR From here, the preprocessing will depend on the rating type
#          So , iterate through each rating type and perform the modelling



N_RATINGS = data.columns.str.contains('rating_').sum()


## This will take some time to complete-- each ratings will be processed separately
## Each model needs to be trained around 5 hours for finding the best hyperparameters
## That means 5 hours * 7 ratings = 35 hours


for TARGET_RATING in range(N_RATINGS):
    
    print("\n Target Rating : ",TARGET_RATING,'\n')
    
    #########################################################################
    #########################################################################
    ### ANCHOR Extract the most important Keywords

    

    how_many_column = range(0,X_text.shape[1]+1) 
    key_word_information = []
    VOCAB_SIZEs = 10_000
    N_FEATURES = 10

    # First, encode the text data locally for each column with tf-idf 
    # Second apply diffusion map for each locally encoded text data

    X_text[X_text == "No inputs"] = ""


    for text_col in how_many_column:
        print("Column Encoding : ",text_col)
        if text_col== X_text.shape[1]:
            N_FEATURES = N_FEATURES*3
            text_col = range(X_text.shape[1])
        text_jointed = pd.DataFrame(X_text.iloc[:,text_col]).fillna("")
        text_jointed = text_jointed.apply(lambda x: " ".join(x), axis=1)
        # Preprocess the text data
        text_jointed = text_jointed.apply(lambda x: preprocess_text(x,
                                                        lower_case=True,
                                                        remove_stop_words=False,
                                                        remove_punctuation=True,
                                                        clear_double_spaces=True,
                                                        change_spelling=True))
            
    
        text_tokens = tf.keras.layers.TextVectorization(
                max_tokens=VOCAB_SIZEs, 
                ngrams=5, 
                output_mode='tf_idf',
                )

        text_tokens.adapt(text_jointed.values)
        ins = tf.keras.layers.Input(shape=(),dtype=tf.string)
        out = text_tokens(ins)
        encoder = tf.keras.Model(ins,out)
        encoded = encoder.predict(text_jointed.values,batch_size=20,verbose=0)

        
        
        encoded = StandardScaler().fit_transform(encoded)
        
        idx = drop_min_correlation(encoded, Y_all[:,TARGET_RATING],  400, 150, .70,verbose=True)
        encoded= encoded[:,idx]
        
        
        print("\n:: Diffusion Map Encoding ::\n")
        encoded = diffusion_map(encoded, n_components=N_FEATURES, t=3,epsilon=None)
        compute_corrcoef(encoded, Y_all, test_train_idx, TARGET_RATING, top_n=50)
        print("\n")
        
        key_word_information.append(encoded)


    ## KEYWORD PREDICTIONS

    X_key_words = np.hstack(key_word_information)
    X_key_words = VarianceThreshold().fit_transform(X_key_words)


    cross_val_forest_score(X_key_words,
                        Y_all,
                        test_train_idx,
                        TARGET_RATING,
                        cv=5,
                        n_estimators=100,
                        max_depth=4,
                        max_samples=.8,
                        random_state=10)




    #########################################################################
    #########################################################################
    ### ANCHOR ALL FEATURES TOGETHER 


    # Combine all features

    # print("Shape of the Text Structure Information",X_STAT.shape)
    print("Shape of the Text Complexity Information",X_LINGO.shape)
    print("Shape of the Keywords Information",X_key_words.shape)
    print("Shape of the Column Embeddings",X_col_embeds.shape)
    print("Shape of the Text Semantic Embeddings",X_encoded.shape)

    ## Global Semantic Pooling with Mean
    X_embeds_mean = X_encoded.mean(axis=1)


    # Unsupervised Feature selection with FRUFS if needed

    # feature_select = int(X_embeds_mean.shape[1]*.9)
    # model = FRUFS.FRUFS(model_c=XGBRegressor(tree_method='gpu_hist', random_state=45),
    #                     k=feature_select,
    #                     n_jobs=-1,
    #                     verbose=0,
    #                     random_state=27)
    # model.fit(X_embeds_mean)
    # X_embeds_mean = model.transform(X_embeds_mean)

    X_embeds_mean.shape


    # Put all feature datasets in a list

    if NLP_FEATURE_STATISTICS:
        feature_datasets = [X_STAT, X_LINGO, X_key_words, X_col_embeds, X_embeds_mean]
    else:
        feature_datasets = [X_LINGO, X_key_words, X_col_embeds, X_embeds_mean]

    # Create a list to hold the reduced datasets
    reduced_datasets = []

    # Iterate over each feature dataset and apply drop_min_correlation
    for i, dataset in enumerate(feature_datasets):
        print("REDUCING DATASET #: ", i)
        idx = drop_min_correlation(
            dataset[test_train_idx],
            Y_all[test_train_idx, TARGET_RATING],
            500,
            500,
            .30)
        # Reduce the dataset size
        reduced_dataset = dataset[:,idx]
        reduced_datasets.append(reduced_dataset)


    ## Combined all features in a 2-D array
    X_combined = np.hstack(reduced_datasets)
    compute_corrcoef(X_combined, Y_all, test_train_idx, TARGET_RATING, top_n=50)

    
    # # PCA with Network Clustering to reduce dimensionality if needed
    
    # X_combined = pca_with_network_cluster(X_combined, 4, threshold=.90)
    # compute_corrcoef(X_combined, Y_all, test_train_idx, TARGET_RATING, top_n=50)

    
    # To normalize get min and max values from the Y 
    MIN_MAX = Y.max().values



    #########################################################################
    #########################################################################
    ### ANCHOR Create Group Weights and Energy Distances 
    
    # Load your dataset
    X_temp = X_combined

    # Standardize your data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_temp)

    # Fine-tuning clustering from 3 to 9 clusters
    best_n_cluster = 0
    best_silhouette = -1
    for n_cluster in range(3, 10):
        print(f"FINE TUNE CLUSTERING: {n_cluster}")
        bisect_means = BisectingKMeans(n_clusters=n_cluster+1, random_state=0).fit(X_scaled)
        try:
            score = silhouette_score(X_scaled, bisect_means.labels_)
        except:
            score = -1
        if score > best_silhouette:
            # print the best silhouette score and the number of clusters
            print("BEST SILHOUETTE SCORE: ", f"{score:.2f}")
            best_silhouette = score
            best_n_cluster = n_cluster+1

    # Apply best clustering found
    clustering = BisectingKMeans(n_clusters=best_n_cluster, random_state=0).fit(X_scaled)    

    # Calculate instance weights and normalize to sum up to 1
    instance_weights = pd.DataFrame(clustering.labels_[~test_train_idx]).value_counts()
    instance_weights_normalized = instance_weights / instance_weights.sum()

    # Apply the twinning library for Multiple K-Fold Cross Validation
    k_fold_ten = twinning.multiplet(X_temp.astype('float64'), 15)

    # Crosstab between K-Folds and train-test index
    # print(pd.crosstab(k_fold_ten, test_train_idx*1)/pd.crosstab(k_fold_ten, test_train_idx*1).sum())

    ## The purpose of this analysis is to make it sure that preprocessing steps did not 
    ## introduce any bias in the data. The following code will show the distribution of
    ## the labels from Bisecting K-Means and the train-test index. If the distribution is
    ## similar, then we can say that the preprocessing steps did not introduce any bias.
    ## Normalized crosstab between labels from Bisecting K-Means and train-test index
    print(pd.crosstab(clustering.labels_, test_train_idx*1) / pd.crosstab(clustering.labels_, test_train_idx*1).sum())



    #########################################################################
    #########################################################################
    ### ANCHOR Create Train and Test Sets for the Ensemble Model and add K-Fold energy distances


    simulation_game_dummies = pd.get_dummies(data.TYPE2).values
    X = X_combined

    X = np.hstack([
        X,
        simulation_game_dummies,
        k_fold_ten.reshape(-1,1)
        ])


    Y_temp = Y_all/MIN_MAX

    X_test = X[~test_train_idx]
    Y_test = Y_temp[~test_train_idx]

    X_train = X[test_train_idx]
    Y_train = Y_temp[test_train_idx]



    #########################################################################
    #########################################################################
    ### ANCHOR Run Ensemble Model With AutoML for Automatic Hyperparameter Search


    # Ensure any existing H2O cluster is shutdown
    for _ in range(2):
        try:
            h2o.cluster().shutdown()
        except:
            pass
        time.sleep(1)


    # Start a new H2O cluster (locally)
    time.sleep(2)
    h2o.init(nthreads=24)
    time.sleep(2)



    # Prepare the train data
    trainFrame = pd.DataFrame(X_train)
    # Rename columns to the format "cols_{i}"
    trainFrame.columns =  [f"cols_{i}" for i in range(trainFrame.shape[1])]
    # Rename the last column to "K_FOLD"
    trainFrame.rename(columns={trainFrame.columns[-1]: "K_FOLD"}, inplace=True)
    # Add the target to the dataframe
    trainFrame['target'] = Y_train[:,TARGET_RATING]

    # Prepare the test data
    testFrame = pd.DataFrame(X_test)
    # Rename columns to the format "cols_{i}"
    testFrame.columns =  [f"cols_{i}" for i in range(testFrame.shape[1])]
    # Rename the last column to "K_FOLD" to maintain consistency with train data
    testFrame.rename(columns={testFrame.columns[-1]: "K_FOLD"}, inplace=True)

    # Print shapes of the train and test frames
    print(f"Train Frame shape: {trainFrame.shape}")
    print(f"Test Frame shape: {testFrame.shape}")



    #########################################################################
    #########################################################################
    ### ANCHOR H2O AutoML for Automatic Hyperparameter Search


    # Convert pandas dataframes to H2O Frames
    train = h2o.H2OFrame(trainFrame)
    test = h2o.H2OFrame(testFrame)

    # Specify predictor variables and response variable
    x = train.columns
    y = "target"
    x.remove(y)  # remove target from predictor list


    exclude_algos = ['DeepLearning']

    # Initialize H2O AutoML
    ## 4 - 5 Hours Hyperparameter Search
    
    aml = H2OAutoML(max_runtime_secs=60*60*5, 
                    exploitation_ratio=.10, 
                    verbosity='info',
                    exclude_algos=exclude_algos
                    )

    # Train the AutoML model
    aml.train(x=x, y=y, training_frame=train, fold_column="K_FOLD")

    # Get AutoML Leaderboard and convert to pandas dataframe
    lb = aml.leaderboard
    lb = lb.as_data_frame()

    # Calculate R2 score for each model in the leaderboard
    r2_score= []

    for model_id in lb['model_id']:
        model = h2o.get_model(model_id)
        cross_val = model.cross_validation_metrics_summary().as_data_frame()
        r2_row = cross_val[cross_val[''].str.contains('r2')]
        r2 = r2_row.iloc[:,1].values
        r2_score.append(np.sqrt(r2))

    # Display R2 scores
    print("*"*100)
    # Show the best 10 models's R2 scores
    print("Best 10 Models' R2 Scores", "\n")
    print(pd.DataFrame(r2_score).head(10))
    # Average R2 score across all models
    print("Average R2 Score", "\n")
    print(pd.DataFrame(r2_score).describe())
    print("*"*100)

    # Clean up memory
    gc.collect()



    #########################################################################
    #########################################################################
    ### ANCHOR Get Final Predictions for the test data

    # Initialize lists to store values
    CV_R2 = []
    CV_sd = []
    predictions = []
    model_name = []

    # Loop over each model in the leaderboard
    for model_id in lb['model_id']:
        # Retrieve the model
        model = h2o.get_model(model_id)
        model_name.append(model.model_id)

        # Extract cross-validation metrics and filter for R2 values
        cross_val = model.cross_validation_metrics_summary().as_data_frame()
        r2_row = cross_val[cross_val[''].str.contains('r2')]
        r2 = r2_row.iloc[:,1].values
        r2_sd = r2_row.iloc[:,2].values

        # Store R2 values and standard deviations
        CV_R2.append(np.abs(r2)**(1/2)*np.sign(r2))
        CV_sd.append(r2_sd)

        # Make predictions on test set and store results
        preds = model.predict(test)['predict'].as_data_frame().values
        predictions.append(preds)



    # Calculate the mean and standard deviation of R2 values across models
    CVs = pd.DataFrame({
        'R': np.array(CV_R2).reshape(-1),
        'R_sd': np.array(CV_sd).reshape(-1),
        'Mean/SD': (np.array(CV_R2)/np.array(CV_sd)).reshape(-1)
    }, index=model_name).sort_values("Mean/SD", ascending=False)

    # Prepare predictions dataframe
    preds = pd.DataFrame(np.swapaxes(np.array(predictions).squeeze(), 0, 1), columns=model_name)

    # If possible, calculate mean predictions for each model type
    for model_type in ['XG', 'GBM', 'DRF', 'DeepLearning']:
        try:
            cols = preds.columns.str.contains(model_type)
            preds[f"{model_type}_MEAN"] = preds.loc[:, cols].mean(axis=1)
        except:
            pass

    # Calculate overall mean and standard deviation of predictions
    preds["MEAN"] = preds.mean(axis=1)
    preds["TOPMEAN"] = preds.iloc[:,:5].mean(axis=1)
    preds["SD"] = preds.std(axis=1)

    # Store final predictions
    X_test_pred = preds
    X_test_pred['response_id'] = data.response_id[~test_train_idx].values
    X_test_pred['TEST_or_DEV'] = data.additional_data[~test_train_idx].values


    #########################################################################
    #########################################################################
    ### ANCHOR Write out the predictions of the test data

    ## Create a prediction folder if it does not exist
    if not os.path.exists("predictions"):
        os.mkdir("predictions")


    path = pathlib.Path(f"predictions/prediction_for_target_{TARGET_RATING}.csv")
    # Save predictions to CSV
    X_test_pred.to_csv(path, index=False)
    print(f"Saved predictions for COLUMN : {TARGET_RATING}")

    # Save cross-validation results to CSV
    # cv_path = pathlib.Path(f"predictions/cv_results_for_target_{TARGET_RATING}.csv")
    # CVs.to_csv(cv_path, index=True)
    # print(f"Saved CV results for COLUMN : {TARGET_RATING}")




#########################################################################
#########################################################################
### ANCHOR Combine each predictions and write out the final predictions


def read_and_preprocess_files(file_list):
    """
    This function takes a list of CSV files as input.
    It reads each CSV file into a pandas DataFrame, sorts them by "response_id",
    resets the index, and extracts the "TOPMEAN" column.
    The processed DataFrames and the "TOPMEAN" columns are stored in separate lists.
    """
    all_predictions, all_predictions_topmean = [], []
    for file in file_list:
        df = pd.read_csv(file)
        df.sort_values("response_id", inplace=True)
        df.reset_index(drop=True, inplace=True)
        all_predictions.append(df)
        all_predictions_topmean.append(df.TOPMEAN.values)

    return all_predictions, all_predictions_topmean


def create_final_predictions(predictions_list, response_ids, test_or_dev):
    """
    This function creates a new DataFrame with the "response_id" column,
    the "TOPMEAN" columns from each prediction DataFrame, and the "TEST_or_DEV" column.
    It also renames and reorders the columns.
    """
    predictions = pd.DataFrame(predictions_list).T
    predictions['response_id'], predictions['TEST_or_DEV'] = response_ids, test_or_dev

    rating_cols = data.columns[data.columns.str.contains("rating")]
    predictions.columns = rating_cols.tolist() + ['response_id', 'TEST_or_DEV']

    # Reordering the columns
    predictions = predictions[['response_id'] + rating_cols.tolist() + ['TEST_or_DEV']]

    return predictions

def write_predictions_to_csv(predictions_df, current_time):
    """
    This function writes the final predictions DataFrame to two CSV files
    after filtering rows based on the "TEST_or_DEV" column value.
    The "TEST_or_DEV" column is also dropped from the DataFrame before writing to CSV.
    The file names are generated based on the current date and time.
    """
    predictions_df[predictions_df.TEST_or_DEV == 0].iloc[:,:-1].to_csv(f"test_predictions_{current_time}.csv", index=False)
    predictions_df[predictions_df.TEST_or_DEV == 1].iloc[:,:-1].to_csv(f"dev_predictions_{current_time}.csv", index=False)


## NOTE: Make it sure that you set the working directory to the "predictions" folder where all the predictions are stored
# Main section of the code
os.chdir("predictions")

# Get all CSV files in the current directory that start with "prediction_for_target_"
files = glob.glob("prediction_for_target_*.csv")

# Read and preprocess all CSV files
all_predictions, all_predictions_topmean = read_and_preprocess_files(files)

# Create the final predictions DataFrame
final_predictions = create_final_predictions(all_predictions_topmean, all_predictions[0].response_id.values, all_predictions[0].TEST_or_DEV.values)

# Write the final predictions DataFrame to CSV files
current_time = time.asctime().replace(" ", "_").replace(":", "_")
write_predictions_to_csv(final_predictions, current_time)