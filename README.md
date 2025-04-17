# Sales_Prediction_using_ML
📈 Sales Prediction Using Machine Learning
🔍 Overview
Sales prediction involves forecasting how much of a product or service will be purchased by customers based on various influencing factors. In this project, we use a machine learning regression model to predict product sales using advertising data.

This project uses the Advertising Dataset from Kaggle, which includes features such as advertising expenditure across different platforms (TV, Radio, Newspaper) and the resulting sales figures.

Problem Statement
Businesses invest in advertising to increase their product sales. However, blindly increasing ad spending does not guarantee improved sales. Understanding which platform (TV, Radio, Newspaper) contributes most to sales can help in optimizing advertising strategies. The goal of this project is to:

Analyze how different media advertising budgets affect sales

Build a regression model to predict sales

Evaluate model performance

Provide actionable insights for marketing strategy


📂 Dataset
Source: Kaggle - Advertising Dataset

Columns:

TV: Advertising budget spent on TV

Radio: Advertising budget spent on Radio

Newspaper: Advertising budget spent on Newspaper

Sales: Units sold

🧰 Technologies Used
Python 🐍

Pandas

NumPy

Matplotlib & Seaborn (EDA & Visualizations)

Scikit-learn (Modeling & Evaluation)


📊 Exploratory Data Analysis (EDA)
Analyzed distributions and relationships between variables using scatter plots and correlation heatmaps

Found that TV and Radio are more strongly correlated with Sales than Newspaper

🧮 Model Building
Used Linear Regression from sklearn.linear_model to build a predictive model.

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)

✅ Model Evaluation
Evaluated model using R² Score to measure accuracy:

from sklearn.metrics import r2_score
r2_score(Y_test, y_test_pred)

🚀 Future Work
Test other regression algorithms (e.g., Ridge, Lasso, Random Forest)

Hyperparameter tuning

Add interaction terms or polynomial features for non-linear patterns
