# Airline Tweets Sentiment Analysis Final Report
## The Problem
Many Twitter users tweet about there travel experiences, particularly with regard to Airlines, and the service they received on particular airlines. This can have an influence on the future purchase decisions of potential passengers, so it is important for Airline management to know how its Airline is perceived on Twitter.

The objective is to build a model than can analyses the sentiment of a tweet. That way a Website can be built around the model which can get a quantative information on an airlines twitter perception over a time period and examine the trend of that perception. Or marketing tools could be built to target users who like an Airline and indicate they would travel again on same airline. Or to quickly reply and flag negative tweets so interventions can provent negative tweets turning into a pile on.

## Model Outcomes
This will be a classification model with the outcome being either a negative, positive, or neutral sentiment. It is be a supervised learning algorithm that is used.

## Data Acquisition
Used the Airline Tweets Sentiment Dataset found on Kaggle, originally from Crowdflower.
![airline_sentiment_pie_charts](https://github.com/user-attachments/assets/a2544d99-fca2-44fc-af0a-4e0853fd72cc)


## Modeling Evaluation
The problem required that I use classification model, and it was supervised. I evaluated the below 8 different models.
 
1.	Logistic Regression
2.	Support Vector Machine (SVM)
3.	Decision Tree
4.	Random Forest
5.	K-Nearest Neighbors (KNN)
6.	Naive Bayes
7.	Neural Networks
8.	Neural Networks (Keras)

## Data Pre-processing
Firstly I tokenize the tweets into individual words. I also convert emojis into text. I then break the words down into their root words (IE Lemmetize) and finally vectorise them. For the airline feature, I one hot encode, and I map the sentiments IE Positive=2, neutral=1, and negative=0. Now the dataset can be used by models that require numeric features.

## Modeling
### Baseline
As specified above, I have decided to try 8 different models. I split the data 4:1 ration between training and test and train the 8 models. Please see results below.
![airline_sentiment_model_results](https://github.com/user-attachments/assets/43290dc9-c018-4cd6-9073-d71f9edf72d4)

| Model         | Train Time  | Train Accuracy | Test Accuracy | Train Precision | Test Precision | Train Recall | Test Recall | Train F1 Score | Test F1 Score |
|---------------|-------------|----------------|---------------|------------------|----------------|--------------|-------------|----------------|---------------|
| LogRegress    | 1.171440    | 0.924309       | 0.871402      | 0.926313         | 0.870846       | 0.924309     | 0.871402    | 0.920955       | 0.860664      |
| KNN           | 0.015734    | 0.407476       | 0.283194      | 0.773582         | 0.692901       | 0.407476     | 0.283194    | 0.442185       | 0.282843      |
| Decision Tree | 5.217317    | 0.999420       | 0.761374      | 0.999419         | 0.756764       | 0.999420     | 0.761374    | 0.999419       | 0.758860      |
| SVC           | 1963.066364 | 0.944045       | 0.779480      | 0.946727         | 0.814459       | 0.944045     | 0.779480    | 0.941794       | 0.727004      |
| Naive Bayes   | 0.000000    | 0.759926       | 0.735840      | 0.812907         | 0.783016       | 0.759926     | 0.735840    | 0.697298       | 0.652304      |
| Random Forest | 34.666027   | 0.999420       | 0.811978      | 0.999421         | 0.816931       | 0.999420     | 0.811978    | 0.999420       | 0.782927      |
| Neural Net    | 244.003658  | 0.999420       | 0.832405      | 0.999420         | 0.821684       | 0.999420     | 0.832405    | 0.999420       | 0.823225      |
| Keras NN      | 76.020786   | 0.999303       | 0.846332      | 0.999304         | 0.845479       | 0.999303     | 0.846332    | 0.999304       | 0.845894      |


Looking at above results, it is clear that Keras Neural Net is the best overall model with high scores across the board. Logistics regression did OK, and was very fast. Random Forest, the Neural Net did well, but not as good as Keras. Also had higher variance. Decison tree clearly overfit, SVC looks like it does too. KNN performed worst, which is a surprise. But the model can probably be improved by balancing the data, and tuning the hyperparameters.

## Improving the Model
### Balancing the Data
As said above, the data is clearly skewed towards the negative, which seems to be affecting the models. So I  use SMOTE, Synthetic Minority Over-sampling Technique — to balance the dataset. The results are below
![airline_sentiment_model_results_bal](https://github.com/user-attachments/assets/3ef3272c-6df7-488b-99c9-4a7f8c36c008)
| Model         | Train Time   | Train Accuracy | Test Accuracy | Train Precision | Test Precision | Train Recall | Test Recall | Train F1 Score | Test F1 Score |
|---------------|--------------|----------------|---------------|------------------|----------------|--------------|-------------|----------------|---------------|
| LogRegress    | 2.726725     | 0.964189       | 0.859796      | 0.964288         | 0.868853       | 0.964189     | 0.859796    | 0.964177       | 0.863037      |
| KNN           | 0.031722     | 0.738675       | 0.264624      | 0.827725         | 0.688690       | 0.738675     | 0.264624    | 0.686180       | 0.240743      |
| Decision Tree | 9.440346     | 0.999724       | 0.752089      | 0.999724         | 0.759415       | 0.999724     | 0.752089    | 0.999724       | 0.755328      |
| SVC           | 1803.385781  | 0.989958       | 0.799443      | 0.990028         | 0.788627       | 0.989958     | 0.799443    | 0.989959       | 0.789271      |
| Naive Bayes   | 0.015840     | 0.921812       | 0.843547      | 0.922207         | 0.847445       | 0.921812     | 0.843547    | 0.921799       | 0.845287      |
| Random Forest | 997.392090   | 0.999724       | 0.843547      | 0.999724         | 0.838498       | 0.999724     | 0.843547    | 0.999724       | 0.831991      |
| Neural Net    | 2358.420714  | 0.999559       | 0.818942      | 0.999559         | 0.807868       | 0.999559     | 0.818942    | 0.999559       | 0.811087      |
| Keras NN      | 137.661974   | 0.999559       | 0.837976      | 0.999559         | 0.830875       | 0.999559     | 0.837976    | 0.999559       | 0.833100      |

In general SMOTE helped the models as the recall and precision scores inproved for all the models. Except for KNN, which clearly is not working for this dataset. SVC looked like it gained most from the balancing. Below gives a visualisation of the comparison

![Model Performance Before and After SMOTE](https://github.com/user-attachments/assets/81c14fdc-6089-4e52-b11c-3c3bb7c25a61)

### Optimizing the Hyperparameters
At this point I pick the best 4 models and see can I optimize the parameters and get a better performance. THe best 4 models are
LogRegress
Naive Bayes
Keras Neural Net
Random Forest 
I use GridSearchCV for all but Keras Neural Net, which does not seem to work well with that library called kerastuner. I trained the model on the balanced training set. See below
![gridsearch_results](https://github.com/user-attachments/assets/61d0d220-db62-40c4-9d48-1382d87cc11e)
| Model         | Train Time   | Train Accuracy | Test Accuracy | Train Precision | Test Precision | Train Recall | Test Recall | Train F1 Score | Test F1 Score |
|---------------|--------------|----------------|---------------|------------------|----------------|---------------|--------------|----------------|----------------|
| LogRegress    | 1465.589591  | 0.995751       | 0.865367      | 0.995761         | 0.865415       | 0.995751      | 0.865367     | 0.995751       | 0.865271       |
| Naive Bayes   | 4863.846278  | 0.957788       | 0.844940      | 0.957848         | 0.844595       | 0.957788      | 0.844940     | 0.957755       | 0.844718       |
| Random Forest | 255.451345   | 0.999724       | 0.844475      | 0.999724         | 0.839394       | 0.999724      | 0.844475     | 0.999724       | 0.833102       |
| Keras NN      | 28731.355840 | 0.999393       | 0.848189      | 0.999393         | 0.842134       | 0.999393      | 0.848189     | 0.999393       | 0.844010       |
Below compares the models before and after tuning the Hyperparameters. As you can see, there is slight improvement.
![Model Performance Before and After Grid](https://github.com/user-attachments/assets/e84e3ad9-5c1d-4ede-97f6-fb8782e43005)

### Overall results of the improvement
As you can see below, Naive Bayes and Random forest benefited most from the Model improvements, mostely by balancing the data.
![Model Performance Before and After Improvement](https://github.com/user-attachments/assets/68ecb46d-3743-4f83-adfa-af0c90e5db1a)

## Comparing against the Marketplace
While I have built a model to do sentiment analysis on Tweets, it is good to compare it's performance to one of the AI engins in the marketplace I chaseo to compare it a gainst OpenAI. I got OPenAI to sentiment analyse a random 2000 samples from the original set, which is roughly the size of the test set of data. The results are in below table, with comparisons of the previous 4 models. I took out the timing metric because I had to slow the OpenAI model down due to cost constraints.

| Model         | Test Accuracy | Test Precision | Test Recall | Test F1 Score |
|---------------|----------------|----------------|-------------|----------------|
| LogRegress    | 0.865367       | 0.865415       | 0.865367    | 0.865271       |
| Naive Bayes   | 0.844940       | 0.844595       | 0.844940    | 0.844718       |
| Random Forest | 0.844475       | 0.839394       | 0.844475    | 0.833102       |
| Keras NN      | 0.848189       | 0.842134       | 0.848189    | 0.844010       |
| **OpenAI**    | **0.9155**     | **0.920884**   | **0.9155**  | **0.917901**   |

Clearly OpenAI Performed best across all metrics against the models I built. Not surprisingly considering the resource gap.

## Conclusion
In order to Analysys tweets OpenAI is clearly the superior sentiment classifier. But in order to realtime scan tweets, it could become expensive. For a more cost effective model, the Logistic Regression model is best.







