#!/usr/bin/env python3
# generate confusion matrix, accuracy bar chart, learning curve
import pickle, numpy as np, matplotlib.pyplot as plt, seaborn as sns, pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# Load
with open('gesture_data.pkl', 'rb') as f:
    X_raw, y_raw = pickle.load(f)
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

X = np.array(X_raw)
y = le.transform(y_raw)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

# 1. Accuracy bar chart
df = pd.read_csv('model_ranking.csv')
plt.figure(figsize=(12,6))
sns.barplot(data=df.head(10), x='Model', y='Test', palette='viridis')
plt.xticks(rotation=45)
plt.title('Top 10 Models by Test Accuracy')
plt.tight_layout()
plt.savefig('accuracy_chart.png')
print("Saved accuracy_chart.png")

# 2. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("Saved confusion_matrix.png")

# 3. Classification report
report = classification_report(y_test, y_pred, target_names=le.classes_)
with open('classification_report.txt', 'w') as f:
    f.write(report)
print("Saved classification_report.txt")

# 4. Learning curve (if possible)
try:
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_test_scaled, y_test, cv=5, train_sizes=np.linspace(0.1,1,10))
    plt.figure()
    plt.plot(train_sizes, np.mean(train_scores,1), 'o-', label='Train')
    plt.plot(train_sizes, np.mean(test_scores,1), 'o-', label='CV')
    plt.legend(); plt.grid(); plt.title('Learning Curve')
    plt.savefig('learning_curve.png')
    print("Saved learning_curve.png")
except: pass

print("\n✅ Evaluation complete. Run main.py for real-time control.")
