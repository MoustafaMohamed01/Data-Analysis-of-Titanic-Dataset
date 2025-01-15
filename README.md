# Data Analysis Project: Titanic Dataset

## Overview
This project involves a comprehensive analysis of the Titanic dataset to uncover insights about passenger demographics, survival rates, and fare distributions. The analysis includes exploratory data analysis (EDA), hypothesis testing, and regression modeling to answer critical questions and highlight significant patterns.

## Features
1. **Descriptive Statistics**
   - Calculated mean, median, mode, variance, and standard deviation for key variables like age and fare.

2. **Data Visualization**
   - **Age Distribution:** Histogram showcasing passenger age distribution.
   - **Fare Distribution:** Histogram illustrating fare variability.
   - **Sex Distribution:** Bar chart comparing the count of male and female passengers.
   - **Survival Distribution:** Bar chart analyzing survival rates.
   - **Age vs. Fare:** Scatter plot to visualize the relationship between passenger age and fare.

3. **Hypothesis Testing**
   - Test of Difference Between Proportions.
   - Independent Two-Samples T-Test.
   - Chi-Square Independence Test.
   - Mann-Whitney U Test.
   - ANOVA (Analysis of Variance).

4. **Regression Analysis**
   - Linear Regression.
   - Logistic Regression.
   - Multiple Regression (with interaction terms).
   - Polynomial Regression.
   - Ridge Regression.
   - Lasso Regression.

## Dataset
- **Source:** Titanic dataset (uploaded).
- **Description:**
  - Passenger information including demographic details, ticket class, survival status, and fares.
  - Contains variables such as `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, and `Embarked`.
- **License:** Public domain (assumed).

## Tools and Technologies
- **Programming Language:** Python
- **Libraries:**
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn
  - SciPy
  - Statsmodels
  - scikit-learn

## Results
1. **Key Insights:**
   - Survival rates varied significantly by sex and passenger class.
   - Younger passengers tended to have lower fares, as highlighted in scatter plots.
2. **Hypothesis Testing Outcomes:**
   - Significant differences were observed in survival rates based on gender (Chi-Square Test).
   - ANOVA revealed variance in fares across different passenger classes.
3. **Regression Analysis:**
   - Logistic regression effectively modeled survival probabilities.
   - Ridge and Lasso regressions provided regularized solutions for predicting fares.


## Contributing
Contributions are welcome! Please fork the repository, make your changes, and create a pull request.

## Acknowledgments
- The dataset was sourced from [Kaggle].
- The dataset link: [https://www.kaggle.com/datasets/yasserh/titanic-dataset].

