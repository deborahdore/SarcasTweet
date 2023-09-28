# SarcasTweet

[SarcasTweet](https://colab.research.google.com/github/deborahdore/SarcasTweet/blob/main/SarcasTweet.ipynb) illustrates how a neural network has been trained to classify italian tweets in
sarcastic or not. The notebook illustrates also all the steps required in order to download tweets from twitter, analyze
them and prepare them for training. At the end, the best model has been integrated into a flask application where it can
be tested by inserting phases in the given form.

## **RUN THE PROJECT**
The Flask app can be run on Docker and the Docker Image can be pulled from the Docker Hub using:<br>
`docker pull deborahdore/sarcastweet`. 

The image can be run using:<br>
`docker run -d -p 5000:5000 deborahdore/sarcastweet`.


## **DATASET**
The dataset was manyually created partially by using tweets and partially by combining pre-existing dataset.
Tweets were manually annotated. 
