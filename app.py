import numpy as np
import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template, jsonify

flask_app = Flask(__name__)
flask_app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')

def load_or_create_model():
    try:
        model = pickle.load(open("model.pkl", "rb"))
        print("✅ Model loaded from model.pkl")
        return model
    except FileNotFoundError:
        print("⚠️ model.pkl not found. Creating new model...")
        try:
            df = pd.read_csv('Crop_recommendation.csv')
            X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
            y = df["label"]
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            pickle.dump(model, open("model.pkl", "wb"))
            print("✅ New model created and saved as model.pkl")
            return model
        except Exception as e:
            print(f"❌ Error creating model: {e}")
            return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

model = load_or_create_model()

@flask_app.route('/')
def home():
    """Home page with form"""
    return render_template("index.html", show_results=False)

@flask_app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if model is None:
        return render_template(
            "index.html",
            prediction_text="Model not loaded. Please try again.",
            show_results=False
        )
    
    try:
        form_data = request.form
        features_list = [
            float(form_data.get('Nitrogen', 0)),
            float(form_data.get('Phosphorus', 0)),
            float(form_data.get('Potassium', 0)),
            float(form_data.get('Temperature', 0)),
            float(form_data.get('Humidity', 0)),
            float(form_data.get('pH', 0)),
            float(form_data.get('Rainfall', 0))
        ]
        
        features = np.array(features_list).reshape(1, -1)
        
        prediction = model.predict(features)
        
        crop_name = str(prediction[0]).capitalize()
        
        top_crops = []
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            class_labels = model.classes_
            
            crops_with_probs = []
            for i, prob in enumerate(probabilities):
                crop = str(class_labels[i]).capitalize()
                crops_with_probs.append((crop, prob))
    
            crops_with_probs.sort(key=lambda x: x[1], reverse=True)
            
            top_count = 0
            for crop, prob in crops_with_probs:
                if crop != crop_name:  
                    top_crops.append((crop, prob))
                    top_count += 1
                    if top_count >= 3:
                        break
        
        return render_template(
            "index.html",
            prediction_text=f"Recommended Crop: {crop_name}",
            top_crops=top_crops,
            show_results=True
        )
        
    except ValueError as e:
        return render_template(
            "index.html",
            prediction_text="Error: Please enter valid numbers",
            show_results=False
        )
    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}",
            show_results=False
        )

@flask_app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
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
  
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            response = {
                'predicted_crop': str(prediction[0]).capitalize(),
                'probabilities': dict(zip([str(c).capitalize() for c in model.classes_], probabilities.tolist()))
            }
        else:
            response = {'predicted_crop': str(prediction[0]).capitalize()}
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@flask_app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@flask_app.route('/crops')
def list_crops():
    """List all available crops"""
    if model and hasattr(model, 'classes_'):
        crops = [str(crop).capitalize() for crop in model.classes_]
        return jsonify({'crops': crops})
    return jsonify({'crops': []})

@flask_app.route('/about')
def about():
    """About page"""
    return render_template("index.html", show_results=False)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    if debug:
        flask_app.run(debug=True, host='0.0.0.0', port=port)
    else:
        from waitress import serve
        print(f"🚀 Starting production server on port {port}")
        serve(flask_app, host='0.0.0.0', port=port)
