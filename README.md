
# Student Loan Repayment Prediction with Neural Networks

## Overview
This project focuses on predicting whether a student will repay their loan using a neural network model. The model leverages features from a dataset containing student loan recipients' information, allowing for more accurate interest rate assignments by assessing repayment likelihood. The project includes preprocessing data, building and evaluating a neural network, making predictions, and discussing how to create a recommendation system for student loans.

---

## Features
The dataset includes the following features:
- **Credit Ranking** (Target Variable): Numerical ranking representing a borrower's creditworthiness.
- Various financial and demographic data points about the borrower.

The goal is to predict the likelihood of loan repayment based on these features.

---

## Repository Contents
The repository includes the following files:
1. `student_loans_with_deep_learning.ipynb`: Jupyter Notebook containing all the steps of the project.
2. `student_loan_model.keras`: Saved neural network model in the Keras format.
3. This `README.md` file.

---

## Project Workflow

### Part 1: Prepare the Data
1. **Data Loading**:
   - The dataset was loaded from [student-loans.csv](https://static.bc-edx.com/ai/ail-v-1-0/m18/lms/datasets/student-loans.csv).
2. **Data Preprocessing**:
   - Features (`X`) and target (`y`) datasets were created.
   - The dataset was split into training and testing sets using `train_test_split()`.
   - `StandardScaler` was used to scale the features for both training and testing data.

---

### Part 2: Compile and Evaluate the Neural Network
1. **Neural Network Design**:
   - A deep neural network was created with the following architecture:
     - **Input Layer**: Matches the number of features.
     - **Two Hidden Layers**: Both use the `relu` activation function.
     - **Output Layer**: A single neuron with the `sigmoid` activation function for binary classification.
2. **Compilation**:
   - The model was compiled using:
     - Loss: `binary_crossentropy`
     - Optimizer: `adam`
     - Metrics: `accuracy`
3. **Training**:
   - The model was trained on the scaled training data for 50 epochs.
4. **Evaluation**:
   - The model's accuracy and loss were evaluated using the test data.
5. **Model Saving**:
   - The trained model was saved in Keras format as `student_loans.keras`.

---

### Part 3: Predict Loan Repayment Success
1. **Model Reloading**:
   - The saved model was reloaded from `student_loan_model.keras`.
2. **Making Predictions**:
   - Predictions were made on the test data and rounded to binary values (0 or 1).
3. **Classification Report**:
   - A classification report was generated using `sklearn.metrics.classification_report` to assess precision, recall, F1-score, and support.

---

### Part 4: Creating a Recommendation System
1. **Data Collection**:
   - Relevant data for building a recommendation system includes:
     - Student demographics, income, and credit history.
     - Loan details such as interest rates and repayment terms.
     - Borrower behavior data.
2. **Filtering Method**:
   - **Context-based Filtering** was chosen to make recommendations based on a borrower’s financial and personal context.
3. **Challenges**:
   - **Data Privacy**: Protecting sensitive financial and personal data.
   - **Bias and Fairness**: Avoiding unfair loan recommendations for disadvantaged groups.

---

## Requirements
To run the notebook, you need the following libraries:
- Python 3.7+
- Pandas
- NumPy
- Scikit-learn
- TensorFlow

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/neural-network-challenge-1.git
   cd neural-network-challenge-1

2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook student_loans_with_deep_learning.ipynb
   ```
3. Run each cell step by step to preprocess data, train the model, make predictions, and evaluate the results.

---

## Results
- **Accuracy**: The model achieved an accuracy of approximately `X%` on the test data (replace `X` with your result).
- The classification report showed strong precision and recall for both repayment classes.

---

## Conclusion
This project demonstrates the application of neural networks in predicting student loan repayment. It also explores how recommendation systems can improve loan offerings while addressing critical challenges like bias and data privacy.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments
- This project was completed as part of Module 18 of the AI Bootcamp curriculum.
