import pandas as pd
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['flower_name'] = pd.Categorical.from_codes(iris.target, iris.target_names) 
# Categorical.from_codes' -> converts number codes (0,1,2) into flower names ('setosa', etc.).
#'Categorical.from_codes' -> pandas advanced method, but .map() does the same thing clearer! 
df

df.head()

print("🌸 IRIS DATASET LOADED! 🌸")
print("="*56)
print("Real flowers measured in 1936 by biologist Ronald Fisher")
print("3 species, 50 flowers each, 4 measurements per flower")
print("="*56)

print("\n📊 Dataset Info:")
print(f"Shape: {df.shape} (rows, columns)")
print(f"\nFlower types (target mapping):")
print("0 = setosa, 1 = versicolor, 2 = virginica")

print("\n🌺 Flower distribution:")
print(df['flower_name'].value_counts())

print("\n💡 Iris flowers: Real category with 260+ species!")
print("   Setosa = Arctic/tough, Versicolor = Colorful, Virginica = Tall/elegant")

df['flower_name'].value_counts()

X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = df['target']

flower_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Keeps the same flower distribution in both sets

print("✅ SPLIT DONE!")
print(f"Train: {len(X_train)} flowers")
print(f"Test:  {len(X_test)} flowers")

print("✅ DATA SPLIT COMPLETE!")
print("="*40)
print(f"Training set: {X_train.shape[0]} flowers")
print(f"Testing set:  {X_test.shape[0]} flowers")

print("\n🌺 Flower distribution in Training set:")
print(pd.Series(y_train).map(flower_names).value_counts())
print("\n🌺 Flower distribution in Testing set:")
print(pd.Series(y_test).map(flower_names).value_counts())

model = LogisticRegression(random_state=42, max_iter=100)
model.fit(X_train, y_train)

print("✅ LOGISTIC REGRESSION TRAINED!")
print("="*40)
print(f"Model converged: {model.n_iter_[0]} iterations")
print(f"Classes learned: {model.classes_}")
print(f"Number of features: {model.n_features_in_}")

# Quick accuracy check
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"\n📊 Training Accuracy:  {train_score:.2%}")
print(f"📊 Testing Accuracy:   {test_score:.2%}")

# Show coefficients (how important each feature is)
print("\n🔍 Feature Coefficients (importance):")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"  {feature:25}: {coef:7.3f}")

print(f"Solver being used: {model.solver}")
print(f"Actual iterations per class: {model.n_iter_}")
print(f"scikit-learn version: {sklearn.__version__}")

y_pred = model.predict(X_test)

print("🎯 PREDICTIONS ON TEST SET")
print("="*50)

y_test_reset = y_test.reset_index(drop=True)
y_pred_series = pd.Series(y_pred)

# Create comparison
test_results = pd.DataFrame({
    'Actual': y_test_reset,
    'Predicted': y_pred_series,
    'Actual_Name': y_test_reset.map(flower_names),
    'Predicted_Name': y_pred_series.map(flower_names)
})

print("First 10 test flowers:")
print(test_results.head(10))

accuracy = accuracy_score(y_test_reset, y_pred)
print(f"\n✅ Test Accuracy: {accuracy:.1%}")

# Check wrong predictions
wrong = test_results[test_results['Actual'] != test_results['Predicted']]
if len(wrong) > 0:
    print(f"\n❌ Wrong predictions ({len(wrong)}):")
    print(wrong)
else:
    print("\n🎉 PERFECT! All predictions correct!")

# Probabilities
print("\n📊 Prediction Probabilities (first 3 flowers):")
probs = model.predict_proba(X_test[:3])
prob_df = pd.DataFrame(probs, columns=[flower_names[c] for c in model.classes_])
print(prob_df.round(3))

y_test

y_test_reset

print("🌸 COEFFICIENTS FOR ALL 3 FLOWER TYPES 🌸")
print("="*60)

# Create a nice table
coef_df = pd.DataFrame(
    model.coef_,
    columns=X.columns,
    index=[flower_names[i] for i in model.classes_]
)

print("Coefficients Table (how each feature affects each flower type):")
print(coef_df.round(3))

print("\n🔍 INTERPRETATION:")
print("Positive = Increases chance of being that flower")
print("Negative = Decreases chance of being that flower")
print("\n💡 Example: High petal length → LESS likely to be Setosa (-2.143)")
print("            High sepal width → MORE likely to be Setosa (+1.158)")

# Show which features matter most for each flower
print("\n🎯 MOST IMPORTANT FEATURE FOR EACH FLOWER:")
for flower_idx, flower_name in enumerate(coef_df.index):
    most_important = coef_df.loc[flower_name].abs().idxmax()
    value = coef_df.loc[flower_name][most_important]
    direction = "increases" if value > 0 else "decreases"
    print(f"  {flower_name:10}: {most_important:20} ({direction} chance)")

flower_names.values()

print("🎭 CONFUSION MATRIX")
print("="*50)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, 
                     index=[f'Actual {name}' for name in flower_names.values()],
                     columns=[f'Pred {name}' for name in flower_names.values()])

print("Confusion Matrix:")
print(cm_df)

# Plot it
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix - Iris Flower Predictions')
plt.ylabel('Actual Flower')
plt.xlabel('Predicted Flower')
plt.tight_layout()
plt.show()

# Detailed classification report
print("\n📋 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=list(flower_names.values())))

print("🎉 FINAL SUMMARY")
print("="*50)

# Final stats
print(f"Dataset: Iris Flowers (n={len(df)})")
print(f"Features: {len(X.columns)} measurements")
print(f"Flower types: {list(flower_names.values())}")
print(f"\nModel: Logistic Regression")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.1%}")
print(f"Iterations needed: {model.n_iter_[0]}/{model.max_iter}")

joblib.dump(model, 'iris_logistic_model.pkl')
print(f"\n💾 Model saved as 'iris_logistic_model.pkl'")

print("\n✅ PROJECT COMPLETE!")
