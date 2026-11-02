import joblib
import pandas as pd

model = joblib.load("model.pkl")

df = pd.read_csv("student_placement_practice_dataset.csv")

X = df.drop(columns=["Student_ID", "Placement_Status"])

predictions = model.predict(X.head(5))

print("Predictions:")

for p in predictions:
    print(p)

assert all(p in ["Placed", "Not Placed"] for p in predictions)

print("Testing completed successfully!")
