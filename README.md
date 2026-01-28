# Portuguese Bank Marketing Classifier Comparison

## Overview

This project implements and compares multiple classification algorithms to predict whether clients will subscribe to a term deposit in the Portuguese banking sector. The analysis covers data preprocessing, feature engineering, model training, evaluation, and optimization recommendations.

---

## Dataset Information

#### Data Share & Composition
Here is a quick statistics on the input data and the split of train/test data used for model training.

| Metric | Value |
|--------|-------|
| **Total Samples** | 41,188 |
| **Training Samples (70%)** | 28,831 |
| **Test Samples (30%)** | 12,357 |
| **Original Features** | 20 |
| **Features After Cleaning** | 9 |

#### Class Distribution

| Class | Count | Percentage |
|-------|-------|-----------|
| **No subscription** | 35,989 | 87.4% |
| **Subscription** | 5,199 | 12.6% |

**Note:** The dataset is imbalanced (88/12 split), which motivated us to pick AUC-ROC as the evaluation metric instead of accuracy.

---

## Data Cleaning & Feature Engineering

#### Features Removed (11 features dropped)

1. **Duration** - Leaked information (only known after call completion)
2. **Day of week** - Low predictive value
3. **Month** - Temporal encoding inefficient
4. **Contact** - Communication type not informative
5. **Macroeconomic indicators** - Multicollinearity:
   - Employment variation rate
   - Consumer price index
   - Consumer confidence index
   - Euribor 3-month rate
   - Number of employees
6. **Pdays** - Previous contact interval with poor predictive power

#### Unknown Values Handling

- Threshold applied: **10% tolerance**
- Dropped columns exceeding threshold: `default` (high 'unknown' ratio)
- Remaining columns with 'unknown' values kept and encoded

#### Features Retained (9 features)

**Numeric Features (4):**
- Age
- Campaign count
- Previous contact count
- Previous outcome

**Categorical Features (5):**
- Job type
- Marital status
- Education level
- Has credit default
- Has housing loan
- Has personal loan

---

## Model Results

### Final Model Performance Comparison

| Rank | Model | Train AUC | Test AUC | Training Time | Improvement vs Baseline |
|------|-------|-----------|----------|---------------|--------------------------|
| **🥇 1st** | **Logistic Regression** | **0.6957** | **0.6896** | 0.067 sec | **+18.45%** |
| 🥈 2nd | K-Nearest Neighbors | 0.8759 | 0.6343 | 0.040 sec | +12.81% |
| 🥉 3rd | Support Vector Machine | 0.7430 | 0.6277 | 187.74 sec | +12.26% |
| 4th | Decision Tree (Tuned) | 0.6684 | 0.6692 | 0.073 sec | +16.41% |
| 5th | Baseline (DummyClassifier) | 0.5014 | 0.5051 | 0.0015 sec | — |

### Key Observations

**✅ Best Model: Logistic Regression**
- **Test AUC-ROC:** 0.6896 (18.45% improvement over baseline)
- **Train-Test Gap:** Only 0.0061 (excellent generalization)
- **Training Speed:** 0.067 seconds (fastest meaningful model)
- **Interpretability:** Feature coefficients available for analysis

**⚠️ Overfitting Issues & Solutions:**
- **KNN:** Train/Test gap of 0.2416 → Indicates memorization; requires n_neighbors tuning
- **Decision Tree:** Originally had 40-point gap → Fixed through hyperparameter tuning:
  - `max_depth=5` (limits tree growth)
  - `min_samples_split=20` (prevents shallow splits)
  - `min_samples_leaf=10` (enforces leaf node size)
  - Result: Gap reduced from 40 to 0.1 points

**⏱️ Training Time Trade-offs:**
- Fastest: Baseline (0.0015 sec)
- Slowest: SVM (187.74 sec due to kernel computation)
- Recommended: Logistic Regression (fast + accurate)

---

## Next steps: Optimization of models

### Phase 1: Hyperparameter Tuning

**For Logistic Regression (Current Best):**
```python
GridSearchCV Parameters:
  - C: [0.001, 0.01, 0.1, 1, 10, 100]        # Regularization strength
  - solver: ['lbfgs', 'liblinear', 'saga']    # Optimization algorithm
  - max_iter: [100, 200, 500, 1000]           # Convergence iterations
```

**For KNN:**
```python
GridSearchCV Parameters:
  - n_neighbors: [3, 5, 7, 9, 11, 15, 20]     # Reduce overfitting
  - weights: ['uniform', 'distance']
  - algorithm: ['ball_tree', 'kd_tree', 'brute']
```

**For SVM:**
```python
GridSearchCV Parameters:
  - C: [0.1, 1, 10, 100]
  - kernel: ['linear', 'rbf', 'poly']
  - gamma: ['scale', 'auto', 0.001, 0.01, 0.1]
```

### Phase 2: Implement Cross-Validation

Replace single train/test split with k-fold cross-validation:
```python
from sklearn.model_selection import RepeatedKFold

cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
```
**Benefits:**
- More robust performance estimates
- Better utilization of limited data
- Confidence intervals for model performance


### Phase 3: Feature Importance Analysis

**Logistic Regression Coefficients:**
```python
# Extract feature importance from fitted model
importance = lr_pipeline.named_steps['lr'].coef_[0]
# Identify top positive/negative drivers of subscription
```

**Decision Tree Importance:**
```python
# Extract from fitted tree
importance = dt_pipeline.named_steps['dt'].feature_importances_
# Identify feature splits with highest information gain
```

## On Deploying the models on Production

### Recommended Model: **Logistic Regression**

**Reasons for picking this model:**
1. **Best Performance:** 0.6896 AUC-ROC (18.45% improvement over baseline)
2. **Excellent Generalization:** Train AUC 0.6957 vs Test AUC 0.6896 (0.6% gap)
3. **Fast Training:** 0.067 seconds (real-time retraining possible)
4. **Interpretable:** Feature coefficients explain subscription drivers
5. **Stable:** Consistent performance across train/test sets

### Next Steps
1. Implement Phase 1 hyperparameter tuning for +1-3% improvement
2. Add cross-validation for robustness assessment
3. Conduct feature importance analysis for business insights
4. Consider ensemble methods (Gradient Boosting) for additional gains
5. Implement SMOTE if recall on positive class needs improvement

