import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data collection n process

print(breast_cancer_dataset)

# loading data to data frames
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)

# first 5 rows of the data frame
data_frame.head()

#adding the 'target' column to the data frame
data_frame['label'] = breast_cancer_dataset.target

data_frame.tail()

#number of rows and col in the dataset
data_frame.shape

#info about data
data_frame.info()

#checking for missing valye 
data_frame.isnull().sum()

# statistical measure
data_frame.describe()

# distrubution of target value
data_frame['label'].value_counts()  

1--> benign
0--> malignant

data_frame.groupby('label').mean()

X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

print(X)

print(Y)

spliting data into training data and testing data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

Model Training

Logistic Regression

model = LogisticRegression()

# traing model
model.fit(X_train, Y_train)

Model Evaluation

Accuracy score

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print('Accuracy on training data = ', training_data_accuracy)

# accuracy of the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Accuracy on test data = ', test_data_accuracy)

predictive system

input_data = (18.25,19.98,119.6,1040,0.09463,0.109,0.1127,0.074,0.1794,0.05742,0.4467,0.7732,3.18,53.91,0.004314,0.01382,0.02254,0.01039,0.01369,0.002179,22.88,27.66,153.2,1606,0.1442,0.2576,0.3784,0.1932,0.3063,0.08368)

# change data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The Breast cancer is Malignant')

else:
  print('The Breast Cancer is Benign')

