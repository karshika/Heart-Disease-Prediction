from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('heart_disease_model.pkl', 'rb'))

# Mapping dictionary for categorical variables
mappings = {
    'sex': {'Male': 1, 'Female': 0},
    'fbs': {'Yes': 1, 'No': 0},
    'restecg': {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2},
    'exang': {'Yes': 1, 'No': 0},
    'slope': {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2},
    'thal': {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form = request.form
        features = []

        # Numeric fields (convert and sanitize)
        numeric_fields = ['age', 'cp', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
        for field in numeric_fields:
            value = form[field].replace(',', '.').strip()
            features.append(float(value))

        # Validate and map categorical inputs
        for field in ['sex', 'fbs', 'restecg', 'exang', 'slope', 'thal']:
            user_input = form.get(field)
            if user_input not in mappings[field]:
                return render_template('index.html', prediction_text=f"⚠️ Please select a valid value for {field.capitalize()}")
            
            mapped_value = mappings[field][user_input]

            # Insert at specific index based on original data order
            if field == 'sex':
                features.insert(1, mapped_value)
            elif field == 'fbs':
                features.insert(4, mapped_value)
            elif field == 'restecg':
                features.insert(6, mapped_value)
            elif field == 'exang':
                features.insert(8, mapped_value)
            elif field == 'slope':
                features.insert(10, mapped_value)
            elif field == 'thal':
                features.append(mapped_value)

        # Make prediction
        final_features = [np.array(features)]
        prediction = model.predict(final_features)

        result = "⬜ Heart Disease Detected" if prediction[0] == 1 else "🟩 No Heart Disease Detected"
        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"⚠️ {str(e)}")
    

if __name__ == '__main__':
    app.run(debug=True)

