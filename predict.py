from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

# Global variables to store model components
model = None
imputer = None
feature_cols = None

def load_model_components():
    """Load all model components"""
    global model, imputer, feature_cols
    
    try:
        print("🔄 Loading model components...")
        
        # Check if model files exist
        required_files = ['lung_cancer_model.pkl', 'imputer.pkl', 'feature_cols.pkl']
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            raise FileNotFoundError(f"Missing model files: {missing_files}. Please run train_model.py first.")
        
        # Load components
        model = joblib.load('lung_cancer_model.pkl')
        imputer = joblib.load('imputer.pkl')
        feature_cols = joblib.load('feature_cols.pkl')
        
        print("✅ Model components loaded successfully!")
        print(f"Features: {feature_cols}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

def validate_input_data(data):
    """Validate input data"""
    errors = []
    
    for col in feature_cols:
        if col not in data:
            errors.append(f"Missing field: {col}")
            continue
            
        try:
            val = int(data[col])
            if col == 'AGE':
                if val < 0 or val > 120:
                    errors.append(f"Age must be between 0 and 120")
            elif col == 'GENDER':
                if val not in [0, 1]:
                    errors.append(f"Gender must be 0 (Female) or 1 (Male)")
            else:
                if val not in [0, 1]:
                    errors.append(f"{col} must be 0 (No) or 1 (Yes)")
        except ValueError:
            errors.append(f"Invalid value for {col}: must be a number")
    
    return errors

@app.route('/')
def home():
    """Home page with prediction form"""
    if not model:
        if not load_model_components():
            return render_template('error.html', 
                                 error="Model not loaded. Please run train_model.py first.")
    
    return render_template("index.html", feature_cols=feature_cols)

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on form input"""
    try:
        if not model:
            if not load_model_components():
                return render_template('error.html', 
                                     error="Model not loaded. Please run train_model.py first.")
        
        # Collect form data
        data = {}
        for col in feature_cols:
            val = request.form.get(col)
            if val is None or val == '':
                return render_template('error.html', 
                                     error=f"Missing value for '{col.replace('_', ' ').title()}'")
            data[col] = val
        
        # Validate input data
        errors = validate_input_data(data)
        if errors:
            return render_template('error.html', error="<br>".join(errors))
        
        # Convert to integers
        for col in feature_cols:
            data[col] = int(data[col])
        
        # Create DataFrame
        input_df = pd.DataFrame([data])
        
        # Apply imputation (same as training)
        input_imputed = pd.DataFrame(
            imputer.transform(input_df), 
            columns=feature_cols
        )
        
        # Make prediction
        prediction_proba = model.predict_proba(input_imputed)[0]
        prediction = model.predict(input_imputed)[0]
        
        # Get probability of positive class
        prob_positive = prediction_proba[1]
        
        # Determine result message
        if prediction == 1:
            result_message = "🛑 HIGH RISK - You may have lung cancer"
            result_class = "high-risk"
        else:
            result_message = "✅ LOW RISK - You may not have lung cancer"
            result_class = "low-risk"
        
        # Risk level based on probability
        if prob_positive >= 0.8:
            risk_level = "Very High"
        elif prob_positive >= 0.6:
            risk_level = "High"
        elif prob_positive >= 0.4:
            risk_level = "Moderate"
        elif prob_positive >= 0.2:
            risk_level = "Low"
        else:
            risk_level = "Very Low"
        
        return render_template(
            "result.html",
            inputs=data,
            probability=round(prob_positive * 100, 1),
            prediction=prediction,
            result_message=result_message,
            result_class=result_class,
            risk_level=risk_level,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
    except Exception as e:
        return render_template('error.html', 
                             error=f"Prediction error: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        if not model:
            if not load_model_components():
                return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        # Validate data
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Check required fields
        missing_fields = [col for col in feature_cols if col not in data]
        if missing_fields:
            return jsonify({'error': f'Missing fields: {missing_fields}'}), 400
        
        # Create DataFrame
        input_df = pd.DataFrame([{col: data[col] for col in feature_cols}])
        
        # Apply imputation
        input_imputed = pd.DataFrame(
            imputer.transform(input_df), 
            columns=feature_cols
        )
        
        # Make prediction
        prediction_proba = model.predict_proba(input_imputed)[0]
        prediction = model.predict(input_imputed)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(prediction_proba[1]),
            'risk_level': 'High' if prediction == 1 else 'Low',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error="Internal server error"), 500

if __name__ == '__main__':
    print("🚀 Starting Lung Cancer Prediction App...")
    
    # Load model components on startup
    if load_model_components():
        print("🌟 App ready! Visit: http://127.0.0.1:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model. Please run train_model.py first.")
        print("Usage: python train_model.py")