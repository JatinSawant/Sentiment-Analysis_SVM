import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (if not already downloaded)
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Load your trained model from the saved file (replace 'your_model_filename.pkl' with the actual filename)
import joblib
pipeline = joblib.load('train_file.pkl')

# Load your new unlabeled dataset from an Excel file
new_df = pd.read_excel('D:\\Mega_Project\\Test_data_excel\\Test_data.xlsx')

# Text preprocessing for the new dataset
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

# Apply preprocessing to the 'text' column in the new dataset
new_df['text'] = new_df['text'].apply(preprocess_text)

# Make predictions using the trained model
new_predictions = pipeline.predict(new_df['text'])

# Add the predicted emotions to the new dataset
new_df['predicted_emotion'] = new_predictions

 # Save the new dataset with predicted emotions to a CSV file
new_df.to_csv('predicted_emotions.csv', index=False)

# Display the new dataset with predicted emotions
print(new_df[['text', 'predicted_emotion']])






