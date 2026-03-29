# ==========================================================
# MACHINE LEARNING ANALYSIS OF ELECTRON AFFINITY
# ACTINIDE COMPLEXES
# Full ML-Chemistry Workflow
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, cross_val_predict, learning_curve
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

from scipy.stats import pearsonr, spearmanr
from numpy.linalg import inv

import xgboost as xgb

sns.set_style("whitegrid")

# ==========================================================
# DATASET
# ==========================================================

metals=['Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr']

atomic_numbers=[90,91,92,93,94,95,96,97,98,99,100,101,102,103]

multiplicity=[1,2,3,4,3,2,1,2,1,2,1,2,1,2]

f_electrons=[0,2,3,4,5,6,7,8,9,10,11,12,13,14]

covalent_radius=[1.79,1.63,1.58,1.55,1.53,1.51,1.50,1.48,1.47,1.46,1.45,1.44,1.43,1.42]

electronegativity=[1.3,1.5,1.38,1.36,1.28,1.3,1.3,1.3,1.3,1.3,1.3,1.3,1.3,1.3]

spin_orbit=[0.20,0.22,0.25,0.26,0.28,0.30,0.31,0.33,0.34,0.36,0.37,0.39,0.40,0.42]

metal_charge=[3]*14

# Electron affinity values
A_prop=[5.53,7.41,7.47,7.05,7.76,7.59,8.04,8.91,8.74,9.06,9.24,9.96,10.34,5.21]

A_acry=[5.95,7.24,7.27,6.89,7.63,7.95,7.83,8.69,8.54,8.85,9.04,9.76,10.14,6.20]

ADCH_prop=[1.11,1.12,1.07,1.06,1.08,1.14,1.38,1.07,1.12,1.15,0.99,1.06,0.99,0.84]
ADCH_acry=[1.10,1.07,0.91,1.06,1.04,0.88,1.52,1.22,1.02,1.11,0.97,1.08,0.99,0.84]

AvgC_prop=[0.42,0.40,0.38,0.37,0.42,0.42,0.41,0.37,0.34,0.33,0.44,0.30,0.25,0.29]
AvgO_prop=[1.00,0.98,0.96,0.95,1.05,1.07,1.02,0.96,0.93,0.89,1.09,0.79,0.75,0.76]

AvgC_acry=[0.42,0.40,0.38,0.37,0.44,0.43,0.40,0.36,0.35,0.33,0.44,0.30,0.26,0.30]
AvgO_acry=[1.01,0.98,0.97,0.94,1.08,1.07,1.03,0.97,0.94,0.90,1.09,0.80,0.76,0.76]

data=[]

for i in range(len(metals)):

    data.append({
        "Metal":metals[i],
        "AtomicNumber":atomic_numbers[i],
        "Multiplicity":multiplicity[i],
        "CovalentRadius":covalent_radius[i],
        "Electronegativity":electronegativity[i],
        "fElectrons":f_electrons[i],
        "SpinOrbit":spin_orbit[i],
        "MetalCharge":metal_charge[i],
        "Ligand":"propionate",
        "ADCH":ADCH_prop[i],
        "Avg_C_M":AvgC_prop[i],
        "Avg_O_M":AvgO_prop[i],
        "EA":A_prop[i]
    })

    data.append({
        "Metal":metals[i],
        "AtomicNumber":atomic_numbers[i],
        "Multiplicity":multiplicity[i],
        "CovalentRadius":covalent_radius[i],
        "Electronegativity":electronegativity[i],
        "fElectrons":f_electrons[i],
        "SpinOrbit":spin_orbit[i],
        "MetalCharge":metal_charge[i],
        "Ligand":"acrylate",
        "ADCH":ADCH_acry[i],
        "Avg_C_M":AvgC_acry[i],
        "Avg_O_M":AvgO_acry[i],
        "EA":A_acry[i]
    })

df=pd.DataFrame(data)

# ==========================================================
# FEATURES
# ==========================================================

X=pd.get_dummies(df.drop(columns=["EA","Metal"]),drop_first=True)
y=df["EA"]

cv=KFold(n_splits=7,shuffle=True,random_state=42)

# ==========================================================
# HYPERPARAMETER TUNING
# ==========================================================

param_grid={
"n_estimators":[300,400,500],
"max_depth":[4,5,6,None],
"min_samples_split":[2,3],
"min_samples_leaf":[1,2]
}

grid=GridSearchCV(
ExtraTreesRegressor(random_state=42),
param_grid,
cv=cv,
scoring="r2",
n_jobs=-1
)

grid.fit(X,y)

best_model=grid.best_estimator_

# ==========================================================
# MODEL COMPARISON
# ==========================================================

models={

"ExtraTrees":ExtraTreesRegressor(random_state=42),

"RandomForest":RandomForestRegressor(n_estimators=400),

"XGBoost":xgb.XGBRegressor(n_estimators=400),

"SVR":SVR(),

"LinearRegression":LinearRegression()

}

results=[]
predictions={}

for name,model in models.items():

    r2_cv=cross_val_score(model,X,y,cv=cv,scoring="r2")

    mae_cv=-cross_val_score(model,X,y,cv=cv,scoring="neg_mean_absolute_error")

    pred=cross_val_predict(model,X,y,cv=cv)

    predictions[name]=pred

    results.append({

    "Model":name,
    "CV_R2":r2_cv.mean(),
    "CV_MAE":mae_cv.mean(),
    "RMSE":np.sqrt(mean_squared_error(y,pred))

    })

results_df=pd.DataFrame(results)

results_df.to_csv("Models_Comparison_Table.csv",index=False)

print(results_df)

# ==========================================================
# FINAL MODEL
# ==========================================================

best_model.fit(X,y)

pred=best_model.predict(X)

table=df[["Metal","Ligand"]].copy()

table["EA_DFT"]=y
table["EA_predicted"]=pred
table["Error"]=pred-y

table.to_csv("ElectronAffinity_Predictions.csv",index=False)

print(table)

# ==========================================================
# STATISTICAL VALIDATION
# ==========================================================

r2=r2_score(y,pred)

rmse=np.sqrt(mean_squared_error(y,pred))

mae=mean_absolute_error(y,pred)

pearson=pearsonr(y,pred)[0]

spearman=spearmanr(y,pred)[0]

print("\nStatistics")
print("R2 =",r2)
print("RMSE =",rmse)
print("MAE =",mae)
print("Pearson =",pearson)
print("Spearman =",spearman)

# ==========================================================
# WILLIAMS PLOT DATA
# ==========================================================
X_num = X.astype(float).values
H = X_num @ inv(X_num.T @ X_num) @ X_num.T
leverage = np.diagonal(H)
h_star = 3 * (X.shape[1] + 1) / X.shape[0]
residuals = (y - pred) / np.std(y - pred)

# ==========================================================
# BOOTSTRAP
# ==========================================================
boot_preds = []
for _ in range(200):
    X_s, y_s = resample(X, y)
    best_model.fit(X_s, y_s)
    boot_preds.append(best_model.predict(X))
pred_std = np.std(np.array(boot_preds), axis=0)

# ==========================================================
# SHAP ANALYSIS (Corrected)
# ==========================================================
explainer = shap.TreeExplainer(best_model)
shap_values = explainer(X)

# ==========================================================
# PLOTS
# ==========================================================

# 1. Parity Plot + Learning Curve
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Parity Plot
axes[0].scatter(y, pred, s=65, alpha=0.85, edgecolors='black', linewidth=0.6)
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2.2, label='Ideal (y = x)')
axes[0].set_xlabel('DFT Electron Affinity (eV)')
axes[0].set_ylabel('Predicted Electron Affinity (eV)')
axes[0].set_title('Parity Plot')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(best_model, X, y, cv=cv, scoring="r2", n_jobs=-1)
axes[1].plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training', linewidth=2.2)
axes[1].plot(train_sizes, test_scores.mean(axis=1), 's-', label='Validation', linewidth=2.2)
axes[1].set_xlabel('Training Set Size')
axes[1].set_ylabel('R² Score')
axes[1].set_title('Learning Curve')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Parity_and_LearningCurve.png', dpi=600, bbox_inches='tight')
plt.show()

# 2a. SHAP Summary Plot
plt.figure(figsize=(7,5))
shap.summary_plot(shap_values.values, X, show=False)
plt.title('SHAP Summary Plot')
plt.tight_layout()
plt.savefig('SHAP_Summary.png', dpi=600, bbox_inches='tight')
plt.show()

# 2b. SHAP Dependence Plot
plt.figure(figsize=(7,5))
shap.dependence_plot("fElectrons", shap_values.values, X, show=False)
plt.title('SHAP Dependence Plot\n(fElectrons vs EA)')
plt.tight_layout()
plt.savefig('SHAP_Dependence.png', dpi=600, bbox_inches='tight')
plt.show()

# 3. Bootstrap + Y-Randomization
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

axes[0].hist(pred_std, bins=12, color='#2E86C1', edgecolor='black', alpha=0.85)
axes[0].set_xlabel('Prediction Uncertainty (std, eV)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Bootstrap Uncertainty Distribution')

r2_random = []
for _ in range(100):
    y_rand = np.random.permutation(y)
    best_model.fit(X, y_rand)
    pred_rand = best_model.predict(X)
    r2_random.append(r2_score(y_rand, pred_rand))

axes[1].hist(r2_random, bins=15, color='#E74C3C', edgecolor='black', alpha=0.85)
axes[1].set_xlabel('R² Score')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Y-Randomization Test')

plt.tight_layout()
plt.savefig('Bootstrap_and_YRandomization.png', dpi=600, bbox_inches='tight')
plt.show()

# 4. Williams Plot
plt.figure(figsize=(6.2, 5.0))
plt.scatter(leverage, residuals, s=65, alpha=0.8, edgecolors='black', linewidth=0.6)
plt.axhline(3, color='red', linestyle='--', lw=1.8)
plt.axhline(-3, color='red', linestyle='--', lw=1.8)
plt.axvline(h_star, color='blue', linestyle='--', lw=1.8, label=f'h* = {h_star:.3f}')
plt.xlabel('Leverage')
plt.ylabel('Standardized Residuals')
plt.title('Williams Plot (Applicability Domain)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Williams_Plot.png', dpi=600, bbox_inches='tight')
plt.show()

# 5. Permutation Feature Importance

perm_importance=permutation_importance(best_model,X,y,n_repeats=10,random_state=42)
fig,ax=plt.subplots(figsize=(4.11,3),dpi=600)
ax.barh(X.columns,perm_importance.importances_mean)
ax.set_title("Permutation Feature Importance")
ax.set_xlabel("Mean Importance")
plt.tight_layout()
plt.savefig("Figure5_perm_importance.png")
plt.show()

# 6. Prediction Error Distribution

errors=pred-y
fig,ax=plt.subplots(figsize=(4.11,3),dpi=600)
ax.hist(errors,bins=10)
ax.set_title("Prediction Error Distribution")
ax.set_xlabel("Predicted - DFT EA")
ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("Figure6_error_distribution.png")
plt.show()

# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(best_model,"Best_ExtraTrees_Model.pkl")

print("\nAnalysis Completed Successfully")