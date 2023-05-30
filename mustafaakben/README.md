# Unleashing the Power of Ensemble Models for SIOP Machine Learning Competition 2023

This repository contains the code and resources for the multi-stage approach to create a text analysis and prediction model for [SIOP Machine Learning Competition 2023](https://eval.ai/web/challenges/challenge-page/1937/overview). This project is designed to process raw text data, extract semantic and feature information, and utilize ensemble models to provide predictions.

If you want to watch the video of my presentation, please click the following link:

[SIOP-Machine Learning Presentation Video](https://www.dropbox.com/s/hvq0ekdn8hbjoe5/SIOP-MachineLearning%20Presentation.mp4?dl=0)

You can access the codes for the project [here](https://github.com/izk8/2023_SIOP_Machine_Learning_Winners/tree/main/mustafaakben/Code)

## Overview of the Proposed Solution

1. Data Cleaning
2. Semantic Embedding
3. Feature Extraction
4. Model Ensembling

### 1. Data Cleaning

As an essential first step in my project, I focus on data cleaning to ensure the quality and accuracy of the data before proceeding to further stages. I divide this process into several sub-steps to achieve a comprehensive and effective cleaning of the raw text data.

1. **White Space Correction**: I leverage open-source white-space correction library on Python to identify and correct white-space errors in the text data. This step ensures that my model can better understand the text and make accurate predictions, extracting the relevant keyword and other information.

2. **Additional Columns Generation**: Using the processed text data, I create six additional columns from the `final_text` using R's `regex` pattern detection to extract essential information for my analysis. These columns enhance context and contribute to a better comprehension of the text data.

3. **Simulation Game Identification**: I identify three different simulation games within the data and code them based on the main player's name (Jamie Pace, Corry Manning, and Carry Stevens). This step allows me to categorize the data according to the respective games, helping the model understand and process the text more effectively. I then encode this information using a masked-encoding schema. 

4. **Handling Systematic Missingness**: Upon noticing a systematic missingness in the data, I perform a cluster analysis using the missing data information with `k-mean` clustering algorithm. This allows me to identify three unique clusters, which I add and use this information as a dummy variables.

5. **Final Dataset Preparation**: After completing the data cleaning and encoding schemes, I compile a dataset with nearly 30 columns, ready for further processing and analysis in subsequent stages of the project.

By cleaning the data, I lay a strong starting point for the following steps of semantic embedding, feature extraction, and model ensembling.

### 2. Semantic Embedding

Semantic embedding plays a crucial role in capturing the meaning and context of the text data, which is vital for text analysis and prediction tasks. In this stage, I employ a deep learning model to obtain embeddings for each column and explore alternative embedding models. The process is divided into the following steps:

1. **Instructor XL Embeds**: I begin by employing a fine-tuned deep learning model, `Instructor XL`, to generate tensor representations for each column. This model is renowned for effectively capturing semantic relationships in text. You can access the Instructor XL model on [Hugging Face](https://huggingface.co/hkunlp/instructor-xl). Once each column is embedded within its semantic space, it results in 3-dimensional tensors, which leads us to the subsequent step.

2. **Global Mean Pooling**: To convert the 3-dimensional tensor into a more manageable 2-dimensional matrix, I apply `global mean pooling` to the column embeddings. This process helps retain the most important features and reduce the computational complexity of the model.

3. **Agglomerative Clustering**: For each column, I compute `agglomerative clusters` (N=10) to obtain the global and local semantic structure of the data. This clustering method further refines the embeddings.

4. **Alternative Embedding Models**: To enhance the quality of the embeddings, I also explore alternative models such as the T5 model from `Tensorflow Hub`. T5 is a powerful pre-trained model that can generate meaningful embeddings for various natural language processing tasks. The T5 model can be found at [TensorFlow Hub](https://tfhub.dev/google/sentence-t5/st5-11b/1). By comparing the performance of the Instructor XL embeds with the T5 model, I can select the best-suited embeddings for the given ratings.

By employing a semantic embedding process, I capture the meaning and context of the text data effectively. This paves the way for efficient feature extraction and accurate predictions using ensemble models in the subsequent stages of the project.

### 3. Feature Extraction

Feature extraction is a critical step in my project, as it helps identify important characteristics and patterns in the text data that influence the prediction outcomes. I use a combination of natural language processing techniques and dimensionality reduction methods to extract relevant features from the cleaned text. The feature extraction process comprises the following steps:

1. **Text Complexity Scores**: I calculate various text complexity scores using natural language processing statistics from `spaCy` such as perplexity, noun-phrase counts, and SMOGs etc. These scores provide insights into the readability and complexity of the text data, which can significantly impact the model's predictions.

2. **TF-IDF Matrix for keywords**: To capture the importance of words and phrases within the text, I create a Term Frequency-Inverse Document Frequency (TF-IDF) matrix with 10,000 words and n-grams (n=5) for each column. The TF-IDF matrix helps identify words and phrases that are unique and informative in the given context, contributing to the model's understanding of the text data. However, there were a little issue with this approach, which leads me to the next step.

3. **Dimensionality Reduction**: The TF-IDF matrix yields a high-dimensional feature space (30 column x 10,000 for each column = 300,000 dimensions), which leads to the curse of dimensionality. To address this issue, I slightly modifed a diffusion map algorithm based on cosine similarity and random walk principles from a spectral analysis perspective. This method helps reduce the dimensions drastically while preserving the local and global feature structure, making it easier for the model to process and analyze the data.

The diffusion map formula is as follows:

$$
\begin{aligned}
K(x_i, x_j) &= \exp \left( -\frac{|x_i - x_j|^2}{\sigma^2} \right),\\
D_{ii} &= \sum_j K(x_i, x_j), \\
P_{ij} &= \frac{K(x_i, x_j)}{D_{ii} D_{jj}}, \\
\alpha_{ij} &= D_{ii} P_{ij} D_{jj}, \\
\lambda_k, \psi_k &= \text{eigenvalues and eigenvectors of } \alpha,\\
Y_k &= \lambda_k \psi_k, \\
\text{Diffusion Map} &= \{Y_1, Y_2, \ldots, Y_n\}.
\end{aligned}
$$

Here, $K(x_i, x_j)$ is the Gaussian kernel, $\sigma$ is the kernel width, $D_{ii}$ is the diagonal matrix with the sum of the kernel values for each point, $P_{ij}$ is the probability transition matrix, $\alpha_{ij}$ is the diffusion matrix, and $\lambda_k$ and $\psi_k$ are the eigenvalues and eigenvectors of the diffusion matrix, respectively.

4. **Bootstrapped Minimum Correlation Feature Selection**: Finally, to further enhance the feature selection process, I employ `bootstrapped minimum correlation feature selection`, which helps identify the most relevant features in the dataset by dropping the least correlated features. This technique is particularly useful for reducing the risk of overfitting and improving the model's generalizability. The bootstrapped minimum correlation feature selection process consists of the following steps:
   1. **Bootstrap Sampling**: I generate multiple bootstrap samples from the original dataset, each containing a random selection of instances with replacement. This process helps create diverse subsets of the data, which can then be used to evaluate the importance of each feature independently.
   2. **Correlation Analysis**: For each bootstrap sample, I calculate the pairwise correlation between the feature and the target value. This step helps identify the relationships between features and assess their relevance in terms of predicting the target variable.
   3. **Feature Ranking**: Based on the correlation analysis, I rank the features according to their importance. The higher the rank of a feature, the more significant its contribution to the model's prediction performance. Then, I dropped the least related features from the data analysis.

By employing these feature extraction techniques, I capture important patterns and characteristics in the text data. This information, combined with the semantic embeddings, enables the ensemble models to provide accurate predictions in the final stage of the project.


# 5. Model Ensembling

After completing the data cleaning, semantic embedding, and feature extraction stages, I move on to the final step of my project, which is model ensembling. In this stage, I combine various machine learning models to make predictions and improve the model's generalizability. The model ensembling process consists of the following steps:

1. **Model Selection**: I used a series of ensemble models for the task, including `XgBoost`, `GradientBoost`, `GLM`, `DRF`, and `Deep Neural Network`. To run the model, I used `h2o` library. These models are chosen based on their ability to capture complex patterns and relationships in the data, and their performance in various text analysis tasks.

2. **K-Fold Cross-Validation**: To assess the performance of the selected models and reduce overfitting, I use a 15-fold cross-validation technique. The k-folds are created using energy distance, which helps minimize variation and increase model generalizability.

The energy distance formula is given as:

$$
E(X, Y) = \sqrt{2 \cdot A(X, Y) - A(X, X) - A(Y, Y)}
$$

where $E(X, Y)$ represents the energy distance between two datasets $X$ and $Y$, and $A(., .)$ is the average Euclidean distance between all pairs of points in the respective datasets.

3. **Hyperparameter Tuning**: To find the optimal hyperparameters for each model, I run an extensive hyperparameter search for about 4 to 10 hours per model. This results in a total of 100 or more different machine learning models for each rating.

4. **Voting Schema**: To make the final predictions, I use a simple voting schema, which combines the predictions of the top-performing models to predict the test scores. This approach leverages the strengths of each model and helps minimize the impact of any single model's weaknesses. I repeat this process for each rating to obtain the most accurate predictions possible.

By employing a ensembling process, I can take advantage of the diverse strengths of multiple machine learning models, resulting in more accurate and reliable predictions for the given text data. This comprehensive approach ensures the effectiveness of the analysis.