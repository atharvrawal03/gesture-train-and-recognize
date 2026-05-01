#!/usr/bin/env python3
# train & compare many classifiers
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
import xgboost as xgb, lightgbm as lgb, catboost as cb, joblib, warnings
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

# Scaling for some models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# All models
models = {
    'KNN': KNeighborsClassifier(5),
    'LogReg': LogisticRegression(max_iter=1000),
    'DecTree': DecisionTreeClassifier(max_depth=10),
    'RF': RandomForestClassifier(100),
    'ExtraTrees': ExtraTreesClassifier(100),
    'GradBoost': GradientBoostingClassifier(100),
    'AdaBoost': AdaBoostClassifier(100),
    'SVM_rbf': SVC(probability=True),
    'SVM_lin': SVC(kernel='linear'),
    'XGB': xgb.XGBClassifier(eval_metric='mlogloss'),
    'LGBM': lgb.LGBMClassifier(verbose=-1),
    'CatBoost': cb.CatBoostClassifier(verbose=0),
    'MLP': MLPClassifier((64,32), max_iter=500),
    'NB': GaussianNB(),
    'QDA': QuadraticDiscriminantAnalysis(),
    'Ridge': RidgeClassifier()
}

results = []
print("\nModel      CV_acc     Test_acc")
print("-" * 40)
for name, mdl in models.items():
    use_scaled = name in ['KNN','LogReg','SVM_rbf','SVM_lin','MLP','Ridge','QDA']
    X_cv = X_train_scaled if use_scaled else X_train
    scores = cross_val_score(mdl, X_cv, y_train, cv=5)
    if use_scaled:
        mdl.fit(X_train_scaled, y_train)
        y_pred = mdl.predict(X_test_scaled)
    else:
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append([name, scores.mean(), scores.std(), acc])
    print(f"{name:12s} {scores.mean():.4f} ± {scores.std():.4f}   {acc:.4f}")

df = pd.DataFrame(results, columns=['Model','CV','Std','Test'])
df = df.sort_values('Test', ascending=False)
df.to_csv('model_ranking.csv', index=False)
print("\n🏆 Model ranking saved to model_ranking.csv")
print(df.to_string(index=False))

# Save best model
best = df.iloc[0]['Model']
for name, mdl in models.items():
    if name == best:
        if name in ['KNN','LogReg','SVM_rbf','SVM_lin','MLP','Ridge','QDA']:
            mdl.fit(X_train_scaled, y_train)
        else:
            mdl.fit(X_train, y_train)
        joblib.dump(mdl, 'best_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(le, 'label_encoder.pkl')
        print(f"\n✅ Best model ({best}) saved as best_model.pkl")
        break
