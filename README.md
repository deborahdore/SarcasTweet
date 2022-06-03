# SarcasTweet

[This Notebook](SarcasTweet_v29.ipynb) illustrates how a neural network has been trained to classify italian tweets in
sarcastic or not. The notebook illustrates also all the steps required in order to download tweets from twitter, analyze
them and prepare them for training. At the end, the best model has been integrated into a flask application where it can
be tested by inserting phases in the given form.

The Flask app can also be run on Docker and the Docker Image can be pulled from the Docker Hub using
`docker pull deborahdore/sarcastweet`. The image can be run using `docker run -d -p 5000:5000 deborahdore/sarcastweet`.
