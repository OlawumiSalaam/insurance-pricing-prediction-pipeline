import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

@app.route('/')
def home():
    """Render the home page with input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests from both form submissions and API calls
    Supports both form-data and application/json content types
    """
    try:
        # Get input data based on content type
        if request.content_type == 'application/json':
            data = request.get_json()
        else:
            data = {
                'age': request.form['age'],
                'sex': request.form['sex'],
                'bmi': request.form['bmi'],
                'children': request.form['children'],
                'smoker': request.form['smoker'],
                'region': request.form['region']
            }
        
        # Create and validate input data
        input_data = CustomData(
            age=int(data.get('age')),
            sex=data.get('sex'),
            bmi=float(data.get('bmi')),
            children=int(data.get('children')),
            smoker=data.get('smoker'),
            region=data.get('region')
        )
        
        # Convert to dataframe and predict
        features = input_data.get_data_as_dataframe()
        pipeline = PredictPipeline()
        prediction = pipeline.predict(features)
        
        # Format response based on content type
        if request.content_type == 'application/json':
            return jsonify({
                'prediction': round(prediction, 2),
                'currency': 'Naira(#)',
                'model_version': '1.0.0'
            })
        else:
            return render_template(
                'index.html',
                prediction_text=f'Estimated Insurance Charge: ${prediction:,.2f}'
            )
            
    except Exception as e:
        error_message = f"Prediction failed: {str(e)}"
        app.logger.error(error_message)
        
        if request.content_type == 'application/json':
            return jsonify({'error': error_message}), 400
        else:
            return render_template(
                'index.html',
                prediction_text=error_message
            ), 400

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'model_ready': True
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)