import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, classification_report
import graphviz

# Load dataset
data = pd.read_csv("bank/bank-full.csv", sep=';')

# Encode categorical columns
label_encoders = {
    col: LabelEncoder().fit(data[col])
    for col in data.select_dtypes(include='object').columns
}
for col, le in label_encoders.items():
    data[col] = le.transform(data[col])

# Features and target
X = data.drop("y", axis=1)
y = data["y"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train decision tree with limited depth
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Create clean Graphviz plot
dot_data = export_graphviz(
    model,
    out_file=None,
    feature_names=X.columns,
    class_names=label_encoders["y"].classes_,
    filled=True,
    rounded=True,
    special_characters=True,
    max_depth=4
)
# Save as PNG image
graph = graphviz.Source(dot_data)
graph.render("clean_decision_tree", format="png", cleanup=True)
print("✅ Clean decision tree saved as 'clean_decision_tree.png'")
