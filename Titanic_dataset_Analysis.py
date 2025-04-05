import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Titanic-Dataset.csv")
print(df.head())

print(df.info())

print(df.describe())

print("\n---------- Mean -------------")
print(df.mean(numeric_only=True))

print("\n---------- Median -------------")

print(df.median(numeric_only=True))

print("\n---------- Mode -------------")
print(df.mode( numeric_only=True))

print("\n---------- Variance ------------- ")
print(df.var(numeric_only=True))

print("\n---------- Standard Deviation -------------")
print(df.std(numeric_only=True))

print("\n------------- Data Visualisation -------------\n")

print("------- Age Distribution (Histogram) ---------\n")
plt.style.use('dark_background')
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'].dropna(), bins=20, kde=True, color='cyan')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('images/age_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("------- Fare Distribution (Histogram) --------\n")
plt.style.use('dark_background')
plt.figure(figsize=(8, 5))
sns.histplot(df['Fare'], bins=20, kde=True, color='green')
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.savefig('images/fare_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("------- Gender Distribution (Bar Chart) ------\n")
plt.style.use('dark_background')
plt.figure(figsize=(8, 5))
ax = sns.countplot(x='Sex', data=df, palette='pastel')
for patch in ax.patches:
    patch.set_alpha(0.8)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.savefig('images/gender_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("------ Survival Distribution (Bar Chart) ------\n")
plt.figure(figsize=(8, 5))
ax = sns.countplot(x='Survived', data=df, palette='Set2')
for patch in ax.patches:
    patch.set_alpha(0.7)
plt.title('Survival Distribution')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.savefig('images/survival_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("---------- Age vs Fare (Scatter Plot) ---------\n")
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df, palette={0: '#FF6347', 1: '#4682B4'}, alpha=0.7)
plt.title('Age vs Fare (Colored by Survival)')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend(title='Survived')
plt.savefig('images/age_vs_fare_survival.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n-------- Hypothesis Test applications ---------\n")
print("---- Test of Difference Between Proportions ---\n")
print("--- Proportions Z-Test (Survival by Gender) ---\n")

from statsmodels.stats.proportion import proportions_ztest
survived_female = df[df['Sex'] == 'female']['Survived'].sum()
print(survived_female)

total_female = df[df['Sex'] == 'female'].shape[0]
print(total_female)

survived_male = df[df['Sex'] == 'male']['Survived'].sum()
print(survived_male)

total_male = df[df['Sex'] == 'male'].shape[0]
print(total_male)

count = [survived_female, survived_male]
print(count)

nobs = [total_female, total_male]
print(nobs)

stat, p_value = proportions_ztest(count, nobs)
print(f"Z-Statistic: {stat:.2f}, P-Value: {p_value}")
if p_value < 0.05:
    print("Significant difference in survival rates between males and females.")
else:
    print("No significant difference in survival rates between males and females.")


print("\n------- Independent Two-Samples T-Test -------\n")
print("-------- T-Test (Fare by Survival) -----------\n")

from scipy.stats import ttest_ind

# Fare data for survivors
fare_survived = df[df['Survived'] == 1]['Fare']

# Fare data for non-survivors
fare_not_survived = df[df['Survived'] == 0]['Fare']

# T-Test
t_stat, p_value = ttest_ind(fare_survived, fare_not_survived, equal_var=False)

print(f"T-Statistic: {t_stat:.2f}, P-Value: {p_value} \n")


if p_value < 0.05:
    print("Significant difference in average fares between survivors and non-survivors.")
else:
    print("No significant difference in average fares between survivors and non-survivors.")


print("\n------- Chi-Square Independence Test -------\n")
print("---- Chi-Square Test (Survival by Class) ---\n")

from scipy.stats import chi2_contingency

# Contingency table for Pclass and Survived
contingency_table = pd.crosstab(df['Pclass'], df['Survived'])

# Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-Square Statistic: {chi2:.2f}, P-Value: {p}\n")

if p < 0.05:
    print("Survival is significantly dependent on passenger class.")
else:
    print("No significant dependency between survival and passenger class.")


print("\n--------- Mann-Whitney U Test --------\n")
print("----- Mann-Whitney U Test (Age by Survival) ----\n")

from scipy.stats import mannwhitneyu
age_survived = df[df['Survived'] == 1]['Age'].dropna()
age_not_survived = df[df['Survived'] == 0]['Age'].dropna()
u_stat, p_value = mannwhitneyu(age_survived, age_not_survived)

print(f"U-Statistic: {u_stat:.2f}, P-Value: {p_value:.4f}\n")
if p_value < 0.05:
    print("Significant difference in age distribution between survivors and non-survivors.")
else:
    print("No significant difference in age distribution between survivors and non-survivors.")


print("\n--------- ANOVA (Analysis of Variance) --------\n")
print("------------- ANOVA (Fare by Class) -----------\n")

from scipy.stats import f_oneway

fare_class_1 = df[df['Pclass'] == 1]['Fare']
fare_class_2 = df[df['Pclass'] == 2]['Fare']
fare_class_3 = df[df['Pclass'] == 3]['Fare']

f_stat, p_value = f_oneway(fare_class_1, fare_class_2, fare_class_3)

print(f"F-Statistic: {f_stat:.2f}, P-Value: {p_value:}\n")
if p_value < 0.05:
    print("Significant difference in mean fares across passenger classes.")
else:
    print("No significant difference in mean fares across passenger classes.")


print("\n------------- Regression analysis ------------\n")
print("-------------- Linear Regression -------------\n")

import statsmodels.api as sm

df = df.dropna(subset=['Age', 'Fare', 'Pclass'])

X = df[['Pclass', 'Age']]
X = sm.add_constant(X)
y = df['Fare']

model = sm.OLS(y, X).fit()
print(model.summary())

print("\n----------- Logistic Regression ----------\n")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

X = df[['Pclass', 'Age', 'Sex']].dropna()
y = df.loc[X.index, 'Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\n----- Multiple Regression (with Interaction) -----\n")

from patsy import dmatrices

formula = "Survived ~ Pclass + Age + Sex + Pclass:Sex"
y, X = dmatrices(formula, df, return_type='dataframe')
logit_model = sm.Logit(y, X).fit()

print(logit_model.summary())

print("\n---------- Polynomial Regression ---------\n")

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = df[['Age']].dropna()
y = df.loc[X.index, 'Fare']

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

print("Coefficients:", poly_reg.coef_)
print("Intercept:", poly_reg.intercept_)

print("\n------------- Ridge Regression -------------\n")

from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

print("Ridge Coefficients:", ridge.coef_)

print("\n------------ Lasso Regression ------------\n")

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

print("Lasso Coefficients:", lasso.coef_)
