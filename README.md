# Airline Tweets Sentiment Analysis Final Report
## The Problem
Many Twitter users tweet about there travel experiences, particularly with regard to Airlines, and the service they received on particular airlines. This can have an influence on the future purchase decisions of potential passengers, so it is important for Airline management to know how its Airline is perceived on Twitter.

The objective is to build a model than can analyses the sentiment of a tweet. That way a Website can be built around the model which can get a quantative information on an airlines twitter perception over a time period and examine the trend of that perception.

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
As specified above, I have decided to try 8 different models. I split the data 4:1 ration between training and test and train the 8 models. Please see results below.


