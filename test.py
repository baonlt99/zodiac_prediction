
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
from sklearn import preprocessing
import pandas as pd
dataset = pd.DataFrame({'Consignor Code':["6402106844","6402106844","6402106844","6402107662","6402107662","6402107662","6408507648"],
                   'Consignee Code': ["66903717","66903717","6404814143","66974631","6404518090","6404518090","6403601344"],
                   'Origin':["DKCPH","DKCPH","DKCPH","DKCPH","DKCPH","DKBLL","DKCPH"],
                   'Destination':["CNPVG","CNPVG","CNPVG","VNSGN","THBKK","THBKK","USTPA"],
                   'Carrier Code':["6402746387","6402746387","6402746387","6402746393","6402746393","6402746393","66565231"]})


#Import the dataset (A CSV file)
#Drop any rows containing NaN values
print(dataset)
#  dataset.dropna(subset=['Consignor Code','Consignee Code','Origin','Destination','Carrier Code'], inplace=True)

#Define our target (What we want to be able to predict)
target = dataset.pop('Destination')

#Convert all our data to numeric values, so we can use the .fit function.
#For that, we use LabelEncoder
le_origin = preprocessing.LabelEncoder()
le_consignor = preprocessing.LabelEncoder()
le_consignee = preprocessing.LabelEncoder()
le_carrier = preprocessing.LabelEncoder()
le_target = preprocessing.LabelEncoder()
target = le_target.fit_transform(list(target))
dataset['Origin'] = le_origin.fit_transform(list(dataset['Origin']))
dataset['Consignor Code'] = le_consignor.fit_transform(list(dataset['Consignor Code']))
dataset['Consignee Code'] = le_consignee.fit_transform(list(dataset['Consignee Code']))
dataset['Carrier Code'] = le_carrier.fit_transform(list(dataset['Carrier Code']))

#Prepare the dataset.
X_train, X_test, y_train, y_test = train_test_split( dataset, target, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

#Print the accuracy score.
print("Accuracy score: {}".format(accuracy_score(y_test, predictions)))

new_input = ["6408507648","6403601344","DKCPH","66565231"]
print(new_input)
fitted_new_input = np.array([le_consignor.transform([new_input[0]])[0],
                                le_consignee.transform([new_input[1]])[0],
                                le_origin.transform([new_input[2]])[0],
                                le_carrier.transform([new_input[3]])[0]])
new_predictions = model.predict(fitted_new_input.reshape(1,-1))

print(le_target.inverse_transform(new_predictions))
