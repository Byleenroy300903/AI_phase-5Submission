import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load the "Fake.csv" dataset
fake_data = pd.read_csv("C:\\Users\\Bylee\\Downloads\\Fake.csv\\Fake.csv")

# Load the "True.csv" dataset
true_data = pd.read_csv("C:\\Users\\Bylee\\Downloads\\True.csv\\True.csv")

# Add labels to distinguish between fake and true news
fake_data['label'] = 0  # 0 for fake news
true_data['label'] = 1  # 1 for true news

# Combine the datasets
combined_data = pd.concat([fake_data, true_data], ignore_index=True)

# Data Preprocessing
combined_data['text'] = combined_data['title'] + " " + combined_data['text']

# Feature Extraction (TF-IDF)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(combined_data['text'])

# Model Selection
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, combined_data['label'], test_size=0.2, random_state=42)

# Logistic Regression Model
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, y_train)

# Model Training (Neural Network)
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(combined_data['text'])
X_train_nn = tokenizer.texts_to_sequences(combined_data['text'])
X_train_nn = pad_sequences(X_train_nn, maxlen=100)

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train_nn, combined_data['label'], epochs=5, batch_size=64)

# Evaluation
# For Logistic Regression
y_pred = logistic_regression_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Logistic Regression Accuracy: {accuracy}")
print(f"Logistic Regression Precision: {precision}")
print(f"Logistic Regression Recall: {recall}")
print(f"Logistic Regression F1-Score: {f1}")
print(f"Logistic Regression ROC-AUC: {roc_auc}")

# For Neural Network
X_test_nn = tokenizer.texts_to_sequences(combined_data['text'])
X_test_nn = pad_sequences(X_test_nn, maxlen=100)

loss, accuracy = model.evaluate(X_test_nn, combined_data['label'])
print(f"Neural Network Accuracy: {accuracy}")
