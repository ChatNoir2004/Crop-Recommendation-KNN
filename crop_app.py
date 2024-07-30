from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

class KNNModel:
    def __init__(self, k=3):
        self.k = k

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def predict(self, x_test):
        predictions = []
        for test_point in x_test:
            distances = np.array([self.euclidean_distance(test_point, train_point) for train_point in self.x_train])
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_indices]
            unique, counts = np.unique(nearest_labels, return_counts=True)
            predicted_label = unique[np.argmax(counts)]
            predictions.append(predicted_label)
        return np.array(predictions)

@app.route('/')
def home():
    return render_template('index.html')  # Your single-page application template

@app.route('/form', methods=["POST"])
def brain():
    try:
        Nitrogen = float(request.form['Nitrogen'])
        Phosphorus = float(request.form['Phosphorus'])
        Potassium = float(request.form['Potassium'])
        Temperature = float(request.form['Temperature'])
        Humidity = float(request.form['Humidity'])
        Ph = float(request.form['ph'])
        Rainfall = float(request.form['Rainfall'])

        values = [Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Ph, Rainfall]

        if Ph > 0 and Ph <= 14 and Temperature < 100 and Humidity > 0:
            # Load the model and labels
            with open('knn_model.pkl', 'rb') as file:
                data = pickle.load(file)
                model = data['model']
                unique_labels = data['labels']

            # Predict using the model
            arr = [values]
            predicted_class_encoded = model.predict(arr)
            predicted_class = unique_labels[predicted_class_encoded[0]]

            # Return prediction as JSON
            return jsonify({'prediction': str(predicted_class)})
        else:
            return jsonify({'error': 'Error in entered values in the form. Please check the values and fill it again.'})

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
