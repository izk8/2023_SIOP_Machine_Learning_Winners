**Team Members**

Ivan Hernandez [https://computationalorganizationalresearch.com](https://computationalorganizationalresearch.com)

Joe Meyer [https://www.linkedin.com/in/meyerjoe152](https://www.linkedin.com/in/meyerjoe152)

Weiwen Nie [https://www.linkedin.com/in/weiwen-nie-255693a7](https://www.linkedin.com/in/weiwen-nie-255693a7)

Andrew Cutler [https://www.linkedin.com/in/andrew-cutler-66791781](https://www.linkedin.com/in/andrew-cutler-66791781)

**MELEE: Conceptual Overview**

Our solution to the problem can be summarized as a multiembedding LSTM ensemble of ensembles, referred to as "MELEE" for simplicity. A key part of MELEE is being able to get different perspectives on what people said, then combining the best of those perspectives, and then combining the specific combinations that work best. Here is the process:

1) Split Data into Train, Holdout, Dev, and Test Sets. Because there is a large amount of ensembling, we need to make sure we set aside data only for ensembling. The LSTM models are trained using 80% of the original trian set. The data used to calibrate the first set of ensemble models that ensemble those LSTMS are the holdout set made from the remaining 20% of the train set. The remaining train set will be used to train the LSTM models. The Dev set is necessary to evaluate the performance of difference ensembles. The Test set is used to make out final predictions

2) Train different LSTM models using the sequences of embedded exercise responses as the inputs. Sentence embeddings offer a way to quantify what text describes along different dimensions. A single embedding model is going to attuned to a subset of specific aspects. So we'll embed each of the in-basket responses with a single embedding model. Then we'll send that sequence of embeddings to a Long Short Term Memory model, which are great at figuring out how a sequence of vectors combine to form an outcome. We'll repeat this process using different embedding models, each being used to train their own LSTM model. These different LSTM models which will each pick up on things that other embedding models do not.

3) Now that we have different LSTM models trained, each with their own perspective, some of those perspectives will capture only a piece of the rules for how the exercises were collectively scored. We'll ensemble the predictions of the different LSTMs to learn how to weight each perspective. Because some perspectives might combine better than others and in different ways, we'll vary the LSTMs that we combine as well as type of ensembling method used

4) Now that we have different ensembles trained, we can further optimize the predictions by combining the ensembles together that work best for specific outcomes. By examining the Dev leaderboard performance of each ensemble, we can know which ensembles are more suited for one outcome over another. We'll have the best set of ensembles for a given outcome make predictions on the test set And then we'll average those predictions together. Those averaged predictions will be submitted to the Test leaderboard.
