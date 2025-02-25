# Customer Churn Prediction App

This project is a **Customer Churn Prediction App** built using the **Flask** framework. The app predicts the likelihood of customers leaving (churning) based on input data. It uses machine learning models to make predictions and provides a simple interface to interact with the model.

## Eaxmple

**Input**

<img src="https://github.com/dhavaldalvi/customer-churn-prediction/blob/main/screenshots/home.png" width="45%" height="45%" />

**Prediction:**

![Screenshot of result page](https://github.com/dhavaldalvi/customer-churn-prediction/blob/main/screenshots/result.png)

---

## Technologies Used

- **Flask**: For building the web application
- **Scikit-learn**: For machine learning and churn prediction
- **Pandas & NumPy**: For data handling and manipulation
- **HTML/CSS**: For creating the user interface

## Requirements
Make sure to have Python installed on your system (Python 3.9 or higher). The app depends on the following Python packages:

- Flask
- Scikit-learn
- Pandas
- NumPy

## Installation

### Prerequisites

Before running the application, make sure you have the following installed:

- **Python 3.9 or higher** 
- **pip** (Python's package installer)

### Or

- **Anaconda** or **Miniconda** (for managing Python environments)

### 1. Go to your desired path where you want to save the application.

### 2. Clone the Repository

Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/dhavaldalvi/customer-churn-prediction.git
```
If you don't have git you can download the project as .zip file. Click on the following link to download

https://github.com/dhavaldalvi/customer-churn-prediction/archive/refs/heads/main.zip

### 3. Go to customer-churn-prediction folder

```bash
cd customer-churn-prediction
```

### 4. Create the Environment

Using `venv` of **Python**. Run the following command in **Command Prompt** or **Terminal**.

```bash
python -m venv myenv
```
### Or

Using `conda` of **Anaconda**. Run following command in **Anaconda Prompt**.

```bash
conda create --name myenv
```
`myenv` is the name of the directory where the virtual environment will be created. You can replace `myenv` with any name you prefer.

### 5. Activating Environment

If using **Python**

In **Windows**
```
.\myenv\Scripts\activate
```
In **MacOS/Linux**
```
source myenv/bin/activate
```
Or if using **Anaconda**

In **Windows**
```
conda activate myenv
```
In **MacOS/Linux**
```
source activate myenv
```

### 6. Run setup.py file

```bash
python setup.py install
```

It will install all required libraries which is listed in 'requirements.txt'.

### 7. Run the Flask App

To start the application, run the following command:

```bash
python app.py
```

It will run multiple models and check their accuracy and select the best model and after it will start the Flask server. By default, the app will be hosted at `http://127.0.0.1:5000`.

---

## Usage

Once the app is running, open your browser and navigate to `http://127.0.0.1:5000`. You will see a form where you can input the parameters. After entering the values, click on the "Submit" button, and the model will predict the result.

---

## Model Details

- **Dataset**: The model was trained using a publicly available Bank Customer Churn Prediction dataset (https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction/data)
- **Machine Learning Algorithm**: Multiple model used for training and the best one is selected.
- **Model File**: The trained model is saved as a `.pkl` file.




