import numpy as np
import os
from flask import Flask, request, render_template, jsonify
import pickle

# Initialize Flask app
flask_app = Flask(__name__)
flask_app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')

# Load the trained model and scaler
try:
    model = pickle.load(open("model.pkl", "rb"))
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


# Add CORS headers for API requests
@flask_app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


@flask_app.route('/')
def home():
    """Home page with form"""
    return render_template("index.html")


@flask_app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Get form data
        form_data = request.form
        features_list = [
            float(form_data.get('Nitrogen', 0)),  # N
            float(form_data.get('Phosphorus', 0)),  # P
            float(form_data.get('Potassium', 0)),  # K
            float(form_data.get('Temperature', 0)),  # temperature
            float(form_data.get('Humidity', 0)),  # humidity
            float(form_data.get('pH', 0)),  # ph
            float(form_data.get('Rainfall', 0))  # rainfall
        ]

        # Convert to numpy array and reshape
        features = np.array(features_list).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        # Get prediction probability
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            class_labels = model.classes_
            top_3_idx = probabilities.argsort()[-3:][::-1]
            top_3_crops = class_labels[top_3_idx]
            top_3_probs = probabilities[top_3_idx]
        else:
            top_3_crops = [prediction[0]]
            top_3_probs = [1.0]

        return render_template(
            "index.html",
            prediction_text=f"🌱 Recommended Crop: {prediction[0]}",
            top_crops=zip(top_3_crops, top_3_probs),
            show_results=True
        )

    except ValueError as e:
        return render_template(
            "index.html",
            prediction_text=f"❌ Error: Please enter valid numbers",
            show_results=False
        )
    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"❌ Error: {str(e)}",
            show_results=False
        )


@flask_app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (for mobile apps or other services)"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()

        features_list = [
            float(data.get('N', data.get('Nitrogen', 0))),
            float(data.get('P', data.get('Phosphorus', 0))),
            float(data.get('K', data.get('Potassium', 0))),
            float(data.get('temperature', data.get('Temperature', 0))),
            float(data.get('humidity', data.get('Humidity', 0))),
            float(data.get('ph', data.get('pH', 0))),
            float(data.get('rainfall', data.get('Rainfall', 0)))
        ]

        features = np.array(features_list).reshape(1, -1)
        prediction = model.predict(features)

        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            response = {
                'predicted_crop': prediction[0],
                'probabilities': dict(zip(model.classes_, probabilities.tolist()))
            }
        else:
            response = {'predicted_crop': prediction[0]}

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@flask_app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


@flask_app.route('/crops')
def list_crops():
    """List all available crops in the model"""
    if model and hasattr(model, 'classes_'):
        return jsonify({'crops': model.classes_.tolist()})
    return jsonify({'crops': []})


if __name__ == "__main__":
    # For production, use waitress or gunicorn instead
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'

    if debug:
        flask_app.run(debug=True, host='0.0.0.0', port=port)
    else:
        # Production server
        from waitress import serve

        print(f"🚀 Starting production server on port {port}")
        serve(flask_app, host='0.0.0.0', port=port)
