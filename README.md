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

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/titanic-data-analysis.git
   ```
2. Navigate to the project directory:
   ```bash
   cd titanic-data-analysis
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Load the Titanic dataset into the `data` folder.
2. Run the analysis scripts to reproduce the results:
   ```bash
   python analysis.py
   ```
3. View the visualizations and analysis outputs in the `outputs` folder.

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

## Folder Structure
```
├── data
│   └── Titanic-Dataset.csv
├── notebooks
│   ├── eda.ipynb
│   ├── hypothesis_testing.ipynb
│   ├── regression_models.ipynb
├── src
│   ├── data_preprocessing.py
│   ├── visualizations.py
│   ├── hypothesis_tests.py
│   ├── regression_analysis.py
├── outputs
│   ├── plots
│   ├── summary_statistics.csv
├── README.md
├── requirements.txt
```

## Contributing
Contributions are welcome! Please fork the repository, make your changes, and create a pull request.

## License
This project is licensed under the MIT License.

## Acknowledgments
- The dataset was sourced from [source, if known].
- Inspired by the [course or tutorial name, if applicable].

