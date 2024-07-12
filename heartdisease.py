import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import Perceptron
from joblib import dump, load
import lime
import lime.lime_tabular

# Suppress all warnings
warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv("F:\\4th sem\\ml\\project\\heart.csv")

# Check if the dataset is balanced or not
class_counts = data['target'].value_counts()
balanced = class_counts[0] == class_counts[1]

# Plot pie chart to visualize class distribution
plt.figure(figsize=(6, 6))
plt.pie(class_counts, labels=['No Heart Disease', 'Heart Disease'], autopct='%1.1f%%', colors=['lightblue', 'lightgreen'])
plt.title("Class Distribution (Balanced: {})".format(balanced))
plt.legend()
plt.show()

# Split features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
dump(scaler, "scaler.joblib")

# Define classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0),
    "XGBoost": XGBClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Support Vector Machine": SVC(),
    "Bagging": BaggingClassifier(),
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Stacking": StackingClassifier(estimators=[('rf', RandomForestClassifier()), ('gb', GradientBoostingClassifier()), ('svm', SVC()), ('knn', KNeighborsClassifier()), ('mlp', MLPClassifier())], final_estimator=LogisticRegression()),
    "Perceptron": Perceptron(),
    "Multi-layer Perceptron": MLPClassifier()
}

# Train and evaluate each classifier
results = {}
confusion_matrices = {}
roc_curves = {}
best_accuracy = 0
best_classifier_name = ""

for name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Compute ROC curve and AUC
    if hasattr(clf, "predict_proba"):
        fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test_scaled)[:,1])
        roc_auc = auc(fpr, tpr)
        roc_curves[name] = {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc}
    else:
        print(f"Warning: Classifier {name} does not support predict_proba, hence ROC curve cannot be plotted.")
    
    results[name] = {"Accuracy": acc, "Classification Report": report}
    confusion_matrices[name] = conf_matrix

    if acc > best_accuracy:
        best_accuracy = acc
        best_classifier_name = name    

# Print results
print("Accuracy Comparison:")
accuracy_df = pd.DataFrame(results).T
accuracy_df.sort_values(by='Accuracy', ascending=False, inplace=True)
plt.figure(figsize=(10, 6))
sns.barplot(data=accuracy_df, x=accuracy_df.index, y='Accuracy', palette='viridis')
plt.xticks(rotation=90)
plt.xlabel("Classifier")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison of Different Classifiers")
plt.tight_layout()
plt.show()

# Print classification report for each classifier
print("\nClassification Report for Each Classifier:")
for name, result in results.items():
    print(f"Classifier: {name}")
    print("Classification Report:")
    print(result["Classification Report"])
    print()

# Combine confusion matrices of all classifiers
print("Confusion Matrices for All Classifiers:")
plt.figure(figsize=(20, 16))
for i, (name, matrix) in enumerate(confusion_matrices.items(), 1):
    plt.subplot(4, 4, i)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
    plt.title(f"{name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# Plot ROC curves
plt.figure(figsize=(10, 6))
for name, curve in roc_curves.items():
    plt.plot(curve["fpr"], curve["tpr"], label=f'{name} (AUC = {curve["roc_auc"]:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.show()

# Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(X_train_scaled, feature_names=X.columns, class_names=['No Heart Disease', 'Heart Disease'])

# Choose an instance to explain (for example, the first instance in the test set)
instance_index = 0
instance = X_test_scaled[instance_index]
true_class = y_test.iloc[instance_index]

# Explain the prediction using Random Forest classifier
explanation_rf = explainer.explain_instance(instance, classifiers['Random Forest'].predict_proba, num_features=len(X.columns))

# Get the explanation as a list of tuples
explanation_list_rf = explanation_rf.as_list(label=1)  # Get the explanation for the predicted label

# Visualize the explanation
features_rf, weights_rf = zip(*explanation_list_rf)
plt.figure(figsize=(10, 6))
plt.barh(features_rf, weights_rf, color='skyblue')
plt.xlabel('Feature Contribution')
plt.ylabel('Feature')
plt.title('Feature Contributions to Prediction (Positive Class) - Random Forest')
plt.show()

# Print results
for name, result in results.items():
    print(name)
    print("Accuracy:", result["Accuracy"])
    print("Classification Report:")
    print(result["Classification Report"])
    print()

# Print confusion matrices
for name, matrix in confusion_matrices.items():
    print(name)
    print("Confusion Matrix:")
    print(matrix)
    print()

best_accuracy = 0
best_classifier = None

# Train and evaluate each classifier
for name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    if acc > best_accuracy:
        best_accuracy = acc
        best_classifier = clf
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    results[name] = {"Accuracy": acc, "Classification Report": report}
    confusion_matrices[name] = conf_matrix

# Print the best classifier and its accuracy
print("Best Classifier:", best_classifier)
print("Best Accuracy:", best_accuracy)

# Train the best classifier on the entire dataset
best_classifier = classifiers[best_classifier_name]
best_classifier.fit(scaler.transform(X), y)

# Save the trained model
dump(best_classifier, "heart_disease_classifier.joblib")
