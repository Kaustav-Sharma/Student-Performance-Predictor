import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, mean_absolute_error, r2_score

# Set visualization style
sns.set_theme(style="whitegrid")

# ==========================================
# 1. DATA PREP (Strictly Unchanged)
# ==========================================
print("Loading dataset...")
df = pd.read_csv(r'c:\Users\kaust\Downloads\student+performance\student-mat.csv', sep=';')

categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop(['G3'], axis=1) 
y_class = np.where(df['G3'] >= 10, 1, 0) 
y_reg = df['G3']

X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.20, random_state=42)
_, _, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# PART A: GRID SEARCH OPTIMIZATION
# ==========================================
print("\nRunning Grid Search to find optimal settings... (This may take a minute)")

# 1. Optimizing Logistic Regression
# Testing different regularization strengths (C) and solvers
log_param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']}
log_grid = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000), log_param_grid, cv=5)
log_grid.fit(X_train_scaled, y_train_class)
best_log_reg = log_grid.best_estimator_

# 2. Optimizing Decision Tree
# Testing different depths, split criteria, and minimum samples per leaf
tree_param_grid = {
    'max_depth': [3, 4, 5, 6, 7], 
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}
tree_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), tree_param_grid, cv=5)
tree_grid.fit(X_train, y_train_class)
best_tree_clf = tree_grid.best_estimator_

# 3. Optimizing Artificial Neural Network
# Testing different network sizes and activation functions
ann_param_grid = {
    'hidden_layer_sizes': [(32,), (64, 32), (32, 16, 8)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.01] # Regularization penalty
}
ann_grid = GridSearchCV(MLPClassifier(solver='adam', max_iter=2000, random_state=42), ann_param_grid, cv=3)
ann_grid.fit(X_train_scaled, y_train_class)
best_ann_clf = ann_grid.best_estimator_

# Generate Predictions with the newly optimized models
y_pred_log = best_log_reg.predict(X_test_scaled)
y_prob_log = best_log_reg.predict_proba(X_test_scaled)[:, 1]

y_pred_tree = best_tree_clf.predict(X_test)
y_prob_tree = best_tree_clf.predict_proba(X_test)[:, 1]

y_pred_ann = best_ann_clf.predict(X_test_scaled)
y_prob_ann = best_ann_clf.predict_proba(X_test_scaled)[:, 1]

# Store Optimized Accuracies
accuracies = {
    'Optimized Logistic Reg': accuracy_score(y_test_class, y_pred_log),
    'Optimized Decision Tree': accuracy_score(y_test_class, y_pred_tree),
    'Optimized ANN': accuracy_score(y_test_class, y_pred_ann)
}

print("\n--- OPTIMAL PARAMETERS FOUND ---")
print(f"Logistic Regression: {log_grid.best_params_}")
print(f"Decision Tree: {tree_grid.best_params_}")
print(f"ANN: {ann_grid.best_params_}")

# ==========================================
# VISUALIZATIONS & MATRICES
# ==========================================

# --- 1. OVERALL ACCURACY BAR CHART ---
plt.figure(figsize=(9, 6))
bars = plt.bar(accuracies.keys(), accuracies.values(), color=['#4C72B0', '#55A868', '#C44E52'])
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval*100:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=12)
plt.ylim(0, 1.1)
plt.title('Post-Optimization Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy Score')
plt.tight_layout()
plt.show()

# --- 2. CONFUSION MATRICES (Side-by-Side) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Optimized Confusion Matrices', fontsize=16, fontweight='bold')
cms = [
    (confusion_matrix(y_test_class, y_pred_log), 'Optimized Logistic Regression'),
    (confusion_matrix(y_test_class, y_pred_tree), 'Optimized Decision Tree'),
    (confusion_matrix(y_test_class, y_pred_ann), 'Optimized ANN')
]
for i, (cm, title) in enumerate(cms):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
    axes[i].set_title(title, fontsize=12)
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')
plt.tight_layout()
plt.show()

# --- 3. ROC CURVES ---
plt.figure(figsize=(8, 6))
fpr_log, tpr_log, _ = roc_curve(y_test_class, y_prob_log)
fpr_tree, tpr_tree, _ = roc_curve(y_test_class, y_prob_tree)
fpr_ann, tpr_ann, _ = roc_curve(y_test_class, y_prob_ann)

plt.plot(fpr_log, tpr_log, label=f'Logistic Reg (AUC = {auc(fpr_log, tpr_log):.3f})', color='blue', lw=2)
plt.plot(fpr_tree, tpr_tree, label=f'Decision Tree (AUC = {auc(fpr_tree, tpr_tree):.3f})', color='green', lw=2)
plt.plot(fpr_ann, tpr_ann, label=f'ANN (AUC = {auc(fpr_ann, tpr_ann):.3f})', color='red', lw=2)
plt.plot([0, 1], [0, 1], 'k--', lw=2) 
plt.title('ROC Curve: Optimized Models', fontsize=14, fontweight='bold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# ==========================================
# PART B: SATISFYING THE REGRESSION REQUIREMENT
# ==========================================
print("\nRunning Regression to satisfy project rubric...")
ann_reg = MLPRegressor(hidden_layer_sizes=ann_grid.best_params_['hidden_layer_sizes'], 
                       activation=ann_grid.best_params_['activation'], 
                       solver='adam', max_iter=2000, random_state=42)
ann_reg.fit(X_train_scaled, y_train_reg)
y_pred_reg = ann_reg.predict(X_test_scaled)

print(f"Regression MAE: {mean_absolute_error(y_test_reg, y_pred_reg):.2f}")
print(f"Regression R2 Score: {r2_score(y_test_reg, y_pred_reg):.2f}")
print("\nOptimization complete!")