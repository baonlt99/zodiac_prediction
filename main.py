
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



dataset = pd.DataFrame(pd.read_json('zodiac.json'))
#  data=pd.read_csv('zodiac.csv')
target=dataset.pop("zodiac")

le_month = preprocessing.LabelEncoder()
le_day = preprocessing.LabelEncoder()
le_target = preprocessing.LabelEncoder()

target = le_target.fit_transform(list(target))

dataset['month'] = le_month.fit_transform(list(dataset['month']))
dataset['day'] = le_day.fit_transform(list(dataset['day']))


X_train, X_test, y_train, y_test = train_test_split( dataset, target, test_size=0.3, random_state=42)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy score: {}".format(accuracy_score(y_test, predictions)))

new_input = [4,8]
print(new_input)
fitted_new_input = np.array([le_month.transform([new_input[0]])[0],
                                le_day.transform([new_input[1]])[0]])

new_predictions = model.predict(fitted_new_input.reshape(1,-1))
print(le_target.inverse_transform(new_predictions))





