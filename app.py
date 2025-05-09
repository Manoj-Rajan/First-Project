from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class_names = ['Setosa', 'Versicolor', 'Virginica']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form.get(f)) for f in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        prediction = model.predict([np.array(features)])[0]
        return render_template('home.html', prediction=class_names[prediction])
    except Exception as e:
        return render_template('home.html', prediction=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")