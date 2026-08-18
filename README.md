# 🎬 IMDb Sentiment Analysis

A machine learning-based sentiment analysis application that predicts whether an IMDb movie review expresses a **positive** or **negative** sentiment.

The project uses **TF-IDF for text feature extraction** and compares multiple machine learning algorithms to identify the best-performing model. The final model is deployed using **Streamlit**.

## 📌 Project Overview

This project uses Natural Language Processing (NLP) and Machine Learning to automatically classify movie reviews into:

- 😊 Positive
- 😞 Negative

The project follows a complete machine learning workflow:

**Data → EDA → Text Preprocessing → TF-IDF → Model Training → Model Comparison → Evaluation → Deployment**

## 🚀 Features

- Exploratory Data Analysis (EDA)
- Duplicate review removal
- HTML tag removal
- Text cleaning and preprocessing
- Lowercase conversion
- TF-IDF feature extraction
- Model comparison
- Sentiment prediction
- Probability-based prediction
- Streamlit web application

## 🧠 Machine Learning Approach

### 1. Text Preprocessing

The following preprocessing steps were performed:

- Removed duplicate reviews
- Removed HTML tags
- Converted text to lowercase
- Removed unnecessary special characters
- Removed extra whitespace

Aggressive preprocessing such as stopword removal, stemming, and lemmatization was avoided to preserve important sentiment-related words such as **"not"**.

### 2. Train-Test Split

The dataset was divided into:

- **80% Training Data**
- **20% Testing Data**

Stratified splitting was used to maintain the class distribution between training and testing datasets.

### 3. TF-IDF Vectorization

TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert textual reviews into numerical features.

Configuration:

```python
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)
Both unigrams and bigrams were used to capture individual words as well as meaningful two-word phrases such as:

amazing
terrible
not good
very enjoyable
🤖 Models Compared

Three machine learning algorithms were evaluated:

Logistic Regression
Multinomial Naive Bayes
Linear SVM
Model Performance
Model	Accuracy	ROC-AUC	F1-Score
Logistic Regression	90.19%	0.9638	0.90
Linear SVM	89.52%	—	0.90
Multinomial Naive Bayes	86.79%	0.9413	0.87
🏆 Final Model

Logistic Regression was selected as the final model because it achieved the highest accuracy and ROC-AUC among the evaluated models.

📊 Final Model Performance

Logistic Regression

Accuracy: 90.19%
ROC-AUC: 0.9638
Precision: ~0.90
Recall: ~0.90
F1-Score: ~0.90
Confusion Matrix
                 Predicted
              Negative Positive


Actual
Negative        4386      554
Positive         419     4558
🖥️ Streamlit Application

The trained Logistic Regression model and TF-IDF vectorizer are saved using Joblib and loaded by the Streamlit application.

The application allows users to:

Enter a movie review
Clean the text
Convert the review into TF-IDF features
Predict the sentiment
Display positive and negative probabilities


🛠️ Technologies Used
Programming Language
Python
Libraries
Pandas
NumPy
Scikit-learn
Joblib
Streamlit
Machine Learning
TF-IDF
Logistic Regression
Multinomial Naive Bayes
Linear SVM

📂 Project Structure
IMDb-Sentiment-Analysis/
│
├── app.py
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── IMDb_Sentiment_Analysis.ipynb
├── README.md
└── requirements.txt

⚙️ Installation

Clone the repository:

git clone <your-github-repository-url>

Navigate to the project directory:

cd IMDb-Sentiment-Analysis

Install the required dependencies:

pip install -r requirements.txt


▶️ Run the Application
Start the Streamlit application:
    streamlit run app.py
The application will open in your browser.

🔮 Future Improvements
Hyperparameter tuning
Advanced NLP models such as BERT
Sentiment confidence visualization
Online deployment
Multilingual sentiment analysis

👩‍💻 Author
Swati
B.Sc. Information Technology
Mumbai University





