#!/usr/bin/env python3
import pickle, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data
with open('gesture_data.pkl', 'rb') as f:
    X_raw, y_raw = pickle.load(f)

X = np.array(X_raw)
le = LabelEncoder()
y = le.fit_transform(y_raw)
print(f"Loaded {len(X)} samples, {len(le.classes_)} classes: {list(le.classes_)}")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models (removed QDA because it fails on high-dim data, add small reg if needed)
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'LogReg': LogisticRegression(max_iter=1000, solver='lbfgs'),
    'DecTree': DecisionTreeClassifier(max_depth=10),
    'RF': RandomForestClassifier(n_estimators=100),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=100),
    'GradBoost': GradientBoostingClassifier(n_estimators=100),
    'AdaBoost': AdaBoostClassifier(n_estimators=100),
    'SVM_rbf': SVC(probability=True, kernel='rbf'),
    'SVM_lin': SVC(kernel='linear'),
    'XGB': xgb.XGBClassifier(n_estimators=100, eval_metric='mlogloss', use_label_encoder=False),
    'LGBM': lgb.LGBMClassifier(n_estimators=100, verbose=-1),
    'CatBoost': cb.CatBoostClassifier(iterations=100, verbose=0),
    'MLP': MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, early_stopping=False),
    'NB': GaussianNB(),
    'Ridge': RidgeClassifier()
}

results = []
print("\nModel          CV_acc      ±       Test_acc")
print("-" * 50)

for name, mdl in models.items():
    use_scaled = name in ['KNN','LogReg','SVM_rbf','SVM_lin','MLP','Ridge']
    try:
        X_cv = X_train_scaled if use_scaled else X_train
        scores = cross_val_score(mdl, X_cv, y_train, cv=5, scoring='accuracy', n_jobs=1)
        mean_cv = scores.mean()
        std_cv = scores.std()

        if use_scaled:
            mdl.fit(X_train_scaled, y_train)
            y_pred = mdl.predict(X_test_scaled)
        else:
            mdl.fit(X_train, y_train)
            y_pred = mdl.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        results.append([name, mean_cv, std_cv, test_acc])
        print(f"{name:12s}   {mean_cv:.4f} ± {std_cv:.4f}     {test_acc:.4f}")
    except Exception as e:
        print(f"{name:12s}   FAILED: {str(e)[:50]}")
        continue

if not results:
    print("No model trained successfully. Check your data.")
    exit(1)

df = pd.DataFrame(results, columns=['Model','CV','Std','Test'])
df = df.sort_values('Test', ascending=False)
df.to_csv('model_ranking.csv', index=False)
print("\n🏆 Model ranking (by test accuracy):")
print(df.to_string(index=False))

# Save best model
best_row = df.iloc[0]
best_name = best_row['Model']
best_cv = best_row['CV']
best_test = best_row['Test']
print(f"\nBest model: {best_name} (CV={best_cv:.4f}, Test={best_test:.4f})")

# Recreate the best model and save
for name, mdl in models.items():
    if name == best_name:
        use_scaled = name in ['KNN','LogReg','SVM_rbf','SVM_lin','MLP','Ridge']
        if use_scaled:
            mdl.fit(X_train_scaled, y_train)
        else:
            mdl.fit(X_train, y_train)
        joblib.dump(mdl, 'best_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(le, 'label_encoder.pkl')
        print(f"✅ Saved {best_name} as best_model.pkl")
        break

print("\nNow run: python3 evaluate.py   and   python3 main.py")
