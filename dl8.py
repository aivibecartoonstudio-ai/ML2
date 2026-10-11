import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv(r"C:\Users\Student\Downloads\reviews.csv")

train_data = df[df["Label"].notna()]
test_data = df[df["Label"].isna()]

X_train = train_data["Review"]
y_train = train_data["Label"]

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

X_test = test_data["Review"]
X_test_vectorized = vectorizer.transform(X_test)

predictions = model.predict(X_test_vectorized)

for review, prediction in zip(X_test, predictions):
    print("Review:", review)
    print("Predicted Sentiment:", prediction)
    print()
