# 🛒 E-commerce Customer Spending Prediction

## 📌 Project Overview

This project predicts **Yearly Amount Spent** by e-commerce customers using **Linear Regression**.

The goal is to understand how customer behavior such as app usage, website usage, session length, and membership duration affects yearly spending.

---

## 📂 Dataset

**Dataset:** Ecommerce Customers Dataset  
**Kaggle Link:** https://www.kaggle.com/datasets/kolawale/focusing-on-mobile-app-or-website

---

## 🎯 Objective

To build a regression model that predicts:

```python
Yearly Amount Spent
```

using:

```python
Avg. Session Length
Time on App
Time on Website
Length of Membership
```

---

## ✅ Project Workflow

```text
Load Dataset
→ Understand Data
→ Check Missing Values
→ Exploratory Data Analysis
→ Visualize Relationships
→ Select Features and Target
→ Split Train/Test Data
→ Train Linear Regression Model
→ Make Predictions
→ Evaluate Model
```

---

## 📊 Exploratory Data Analysis

### Dataset Preview

The dataset contains customer details such as email, address, avatar, app usage, website usage, membership length, and yearly spending.

### Pairplot Analysis

Pairplot was used to understand relationships between numerical columns.

```python
sns.pairplot(
    customers,
    kind="scatter",
    plot_kws={"alpha": 0.3},
    diag_kws={"bins": 40}
)
```

![Pairplot](images/pairplot.png)

---

## 🔍 Strong Relationship Found

`Length of Membership` shows a strong positive linear relationship with `Yearly Amount Spent`.

```python
sns.lmplot(
    x="Length of Membership",
    y="Yearly Amount Spent",
    data=customers,
    scatter_kws={"alpha": 0.3}
)
```

![Length of Membership vs Yearly Amount Spent](images/membership-vs-spending.png)

---

## 🧠 Why This Model?

Linear Regression is used because the target variable is continuous.

Here, the model predicts how much a customer may spend in a year based on numerical customer behavior features.

---

## ✂️ Train Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)
```

### Why train/test split?

We train the model on one part of the data and test it on unseen data.

This helps check whether the model really learned the pattern instead of memorizing the dataset.

---

## 🤖 Model Training

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

### Why `fit()`?

`fit()` trains the model.  
It learns the relationship between input features and the target value.

---

## 🔮 Prediction

```python
pred = model.predict(X_test)
```

### Why `predict()`?

After training, `predict()` is used to estimate spending for new/unseen customer data.

---

## 📈 Actual vs Predicted

```python
sns.scatterplot(x=y_test, y=pred)
plt.ylabel("Prediction")
plt.title("Yearly Amount Spent vs Model Prediction")
```

![Actual vs Predicted](images/actual-vs-predicted.png)

The points are close to a straight line, which shows the model predictions are very close to actual values.

---

## 📊 Model Evaluation

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print(f"MAE      : {mean_absolute_error(y_test, pred):.2f}")
print(f"MSE      : {mean_squared_error(y_test, pred):.2f}")
print(f"R2 Score : {r2_score(y_test, pred):.2f}")
```

| Metric | Value |
|---|---:|
| MAE | 8.43 |
| MSE | 103.92 |
| R² Score | 0.98 |
| Train R² | 0.985 |
| Test R² | 0.981 |

---

## ✅ Result Interpretation

The model achieved an **R² score of 0.98**, which means it explains about **98% of the variation** in yearly customer spending.

The training and testing scores are very close:

```text
Train R² = 0.985
Test R²  = 0.981
```

This shows the model is performing well and does not show major overfitting.

---

## 🧾 Coefficients

```python
coef = pd.DataFrame(model.coef_, X.columns, columns=["coef"])
print(coef)
```

| Feature | Coefficient |
|---|---:|
| Avg. Session Length | 25.72 |
| Time on App | 38.59 |
| Time on Website | 0.46 |
| Length of Membership | 61.67 |

### Coefficient Meaning

- `Length of Membership` has the highest impact.
- `Time on App` is more important than `Time on Website`.
- `Time on Website` has very low impact compared to other features.

---

## 📌 Final Conclusion

This Linear Regression model performs very well for predicting yearly customer spending.

Main insights:

- Longer membership strongly increases yearly spending.
- App usage has more impact than website usage.
- The model gives highly accurate predictions with R² around 0.98.

---

## 🚀 Future Improvements

- Add more customer behavior features
- Compare with Polynomial Regression
- Compare with Decision Tree Regressor
- Compare with Random Forest Regressor
- Build a Streamlit web app for prediction

---

## 👨‍💻 Author

**Arulprasath D**

Machine Learning | Data Science | Python
