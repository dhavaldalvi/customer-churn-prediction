from flask import Flask, render_template, request

from src.customer_churn_prediction.pipelines.prediction_pipeline import CustomData, PredictPipeline
from src.customer_churn_prediction.pipelines.training_pipeline import TrainingPipeline

app = Flask(__name__)

# Initiating training pipeline
TrainingPipeline().initiate_training_pipeline()

# Home page
@app.route('/')
def home():
    return render_template('home.html')

# Prediction page
@app.route('/predict', methods=['POST'])
def predict():
    # Collecting data from html form
    data = CustomData(
        credit_score = request.form.get('CreditScore'),
        geography = request.form.get('Geography'),
        gender = request.form.get('Gender'),
        age = request.form.get('Age'),
        tenure = request.form.get('Tenure'),
        balance = request.form.get('Balance'),
        num_of_products = request.form.get('NumOfProducts'),
        has_credit_card = request.form.get('HasCrCard'),
        is_active_member = request.form.get('IsActiveMember'),
        estimated_salary = request.form.get('EstimatedSalary'),
    )

    # Transforming the collected data to dataframe
    prediction_data = data.data_to_dataframe()
    # Predicting using collected data
    predict_pipeline = PredictPipeline()
    # Storing the result of prediction
    prediction = predict_pipeline.predict(prediction_data)[0]
    # Storing the probability of that result
    probability = predict_pipeline.predict(prediction_data)[1]
    
    # Condtion to give the result depending on the class
    if prediction[0] == 0:
        result = 'No'
        probability = probability[0]
    else:
        result = 'Yes'
        probability = probability[1]

    return render_template('result.html', result=result, probability = round(probability,2))


if __name__ == '__main__':
   app.run(debug=False)
