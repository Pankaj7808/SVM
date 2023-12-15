import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

df = pd.read_csv('diabetes.csv')

x = df.drop('Outcome', axis=1)
y = df['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  # Correctly split the data

classifier = SVC(kernel="linear")
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

print('Confusion Matrix :')
print(confusion_matrix(y_test, y_pred))  # Provide y_test and y_pred as arguments

print('Accuracy Score')
print(accuracy_score(y_test, y_pred))

print('Classification Report : ')
print(classification_report(y_test, y_pred))  # Import classification_report correctly
