import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import make_pipeline
import joblib
from sklearn.metrics import accuracy_score

# Download NLTK resources (if not already downloaded)
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Load your dataset from an Excel file (replace 'C:\\full\\path\\to\\your\\emo_data.xlsx' with the actual absolute file path)
df = pd.read_excel('D:\Mega_Project\Train_data_excel\Train_dataset .xlsx')

# Text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Lemmatization
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# Apply preprocessing to the 'Text' column
df['Text'] = df['Text'].apply(preprocess_text)

# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(df['Text'], df['Emotion'], test_size=0.2, random_state=42)

# Assuming 'Emotion' is the target column
# If your target column has a different name, replace 'Emotion' with the actual name
# Also, replace 'df['Emotion']' with 'df['Text']' in the next line
X = df['Text']
y = df['Emotion']

# Build a pipeline with TF-IDF vectorizer and SVM classifier
pipeline = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))

# Perform GridSearchCV to find the best parameters
param_grid = {'tfidfvectorizer__max_features': [1000, 5000, 10000],
              'svc__C': [0.1, 1, 10]}
grid = GridSearchCV(pipeline, param_grid, cv=3)
grid.fit(train_data, train_labels)

# Get the best model from GridSearchCV
best_model = grid.best_estimator_

# Make predictions on the test set
predictions = best_model.predict(test_data)

# Evaluate the model
accuracy = accuracy_score(test_labels, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Display classification report
print('Classification Report:\n', classification_report(test_labels, predictions))

# Train the model on the entire dataset
pipeline.fit(X, y)

# Example: Predict the emotion for a new sentence
new_sentence = "In the quiet aftermath, shattered fragments of trust lingered, leaving behind a silence heavy with the weight of unspoken grievances."
# Apply the same preprocessing to the new sentence
preprocessed_sentence = preprocess_text(new_sentence)
# Make predictions using the trained model
prediction = pipeline.predict([preprocessed_sentence])

# Display the predicted emotion
print(f'The predicted emotion for the sentence "{new_sentence}" is: {prediction[0]}')

#joblib
joblib.dump(pipeline, 'train_file.pkl')
