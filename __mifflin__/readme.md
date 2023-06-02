team \__mifflin__\: Ammar Ansari 

My approach involved the following:

1. Impute the mean for missing rating values in the training set. 
2. Clean text of email prompts using the stringr library in R.
3. Concatenate all participant text responses into a single column. 
4. Correct misspelled words and words missing spaces in between them using a script I wrote combining the pyenchant and wordninja python libraries. 
5. "Cluster" participants into groups depending on which questions each participant received to ensure each cluster is reasonably similar to the other rows in the cluster in terms of text exercise questions received.
5. Obtain the unigrams/bigrams vector for the new response column for the train + test dataset for each cluster using the spaCy library.
6. Append a column of cosine similarities of this unigrams/bigrams vector for each row (so a new column for the cosine similarity to every other row, for each row). 
7. Repeat steps 5 and 6 but with the distilbert text embeddings of the response column vector rather than unigrams/bigrams. 
8. Using various machine learning algorithms depending on cluster and rating type (either ridge regression, random forests, or bagged trees generally), select the best performing algorithm to predict that specific rating with these cosine similarity columns as predictors, according to my test mean_r functions and the holdout set I split the train set into to evaluate, and create predictions using the 'best' algorithm for each rating within each cluster's actual test set data.
9. Rbind all the predictions for all three clusters together into one predictions dataset.
10. Using the quantile function in R, round the predicted values so that they fall along the same distribution as the test set in terms of observed outcome values, instead of rounding to the nearest whole number (e.g., the lowest X percent of predicted values would fall into the same bucket as the lowest value observed in the training set such that the training set and the test set predictions have the same percentage of values falling in that value. Essentially, the histogram of observed values should look the same after this procedure for the training set and the predicted values on the test set.) 


My final solution combined three models: 

For 5 out of the 7 ratings, I used the predictions with the unigrams/bigrams cosine similarity scores and distilbert embeddings cosine similarity scores as predictors, as described in the steps above. 

For rating_interprets_information, I additionally added the columns for cosine similarities of the TFIDF vectors for each cluster, in the same way as for unigrams/bigrams and distilbert embeddings as described above. 

For rating_involves_others, I used a different model that computed each cosine similarity over the whole data set, and also included the actual distilbert embeddings, as well as the TFIDF scores for each word instead of the cosine similarities. In this model I additionally included counts of the words "please" and "thank", and the overall and mean length of participant responses as predictors as well, before separating/clustering them into only 2 groups after all these predictors were added (unlike all the previous models, in which participants were grouped/clustered into 3 groups before any NLP analysis was performed). 

More detailed descriptions can be found in the jupyter notebook file containing all the code as well. 
