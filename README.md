Fake News Detection Model
This repository contains code for a Fake News Detection Model developed using Natural Language Processing (NLP) techniques and machine learning algorithms. The model distinguishes between genuine and fake news articles based on their textual content.

Dependencies
To run the code, you'll need the following dependencies:

Python (3.x)
Libraries:
pandas
scikit-learn
TensorFlow
Keras
You can install the required libraries via pip using the following command:
pip install pandas scikit-learn tensorflow


Dataset
The repository includes two datasets:

Fake.csv: Contains fabricated news articles.
True.csv: Contains legitimate news articles.
Ensure the CSV files are located in the root directory of the project before running the code.

Running the Code
Clone the Repository:
git clone https://github.com/Byleenroy300903/AI_phase-5Submission.git
cd fake-news-detection


Set Up the Environment:

Ensure you have Python installed.
Install the required dependencies as mentioned above.
Run the Code:

Execute the provided Python script (filename.py) that contains the code snippet shared in this repository. Ensure the necessary datasets are in the correct paths.
Results:

The code generates evaluation metrics for both the Logistic Regression and Neural Network models. The metrics include accuracy, precision, recall, F1-score, and ROC-AUC for the Logistic Regression model, and accuracy for the Neural Network model.
