# Disaster-Response-Pipeline
This project holds repository that contains code for application which can be used by employees during a disaster event (e.g. an earthquake or hurricane), to be able to classify the messages into several categories, in order that the message can be directed to the appropriate aid agencies.  The app uses a ML model to categorize any new messages received, and the repository also contains the code used to train the model and to prepare any new data-sets for model training purposes.


# Required libraries
    1. nltk 3.3.0
    2. numpy 1.15.2
    3. pandas 0.23.4
    4. scikit-learn 0.20.0
    5. sqlalchemy 1.2.12
    
# Motivation

   In this project, It will provide disaster responses to analyze data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.

   This project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.


# File Descriptions

## ETL Pipeline Data cleaning pipeline contained in data/process_data.py:

    Loads the messages and categories datasets
    Merges the two datasets
    Cleans the data
    Stores it in a SQLite database
    ML Pipeline Machine learning pipeline contained in model/train_classifier.py:

## Loads data from the SQLite database
    Splits the dataset into training and test sets
    Builds a text processing and machine learning pipeline
    Trains and tunes a model using GridSearchCV
    Outputs results on the test set
    Exports the final model as a pickle file
## Flask Web App

    It displays results in a Flask web app which is deployed on Heroku.
    
 ## Instructions:
  Run the following commands in the project's root directory to set up your database and model.

  To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
  To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/model.p
  Run the following command in the app's directory to run your web app. python run.py

  Go to [http://0.0.0.0:3001/](https://view6914b2f4-3001.udacity-student-workspaces.com/) for accesing the UI of the app
    
    
 ## Acknowledgements
    I wish to thank Figure Eight for dataset, and thank Udacity for advice and review.
