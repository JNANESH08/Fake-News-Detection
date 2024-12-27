import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, render_template, request

app = Flask(__name__)

# Function to train and save the model and vectorizer
def train_model():
    data = pd.read_csv("dataset.csv")
    x = np.array(data['title'])
    y = np.array(data['label'])

    cv = CountVectorizer()
    x = cv.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(x_train, y_train)

    # Save the model and vectorizer
    joblib.dump(model, 'model.pkl')
    joblib.dump(cv, 'vectorizer.pkl')

# Ensure the model is trained and saved before the first request
train_model()

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/pred', methods=["GET", "POST"])
def pred():
    if request.method == "POST":
        try:
            d = request.form['data']

            # Load the model and vectorizer
            model = joblib.load('model.pkl')
            cv = joblib.load('vectorizer.pkl')

            news_headline = d
            data = cv.transform([news_headline]).toarray()

            r = model.predict(data)

            return render_template('main.html', data=[d, r[0]])
        except Exception as e:
            return render_template('main.html', data=["Error occurred", str(e)])
    return render_template('main.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
