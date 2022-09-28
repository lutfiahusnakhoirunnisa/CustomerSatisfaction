import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

from google.colab import drive
drive.mount('/content/drive')

#Data importing
dataset = pd.read_csv('/content/drive/MyDrive/train.csv')

#Data Cleansing
#checking missing value for each feature  
print('Checking missing value for each feature:')
print(dataset.isnull().sum())
#Counting total missing value
print('\nCounting total missing value:')
print(dataset.isnull().sum().sum())
#Counting the duplicated data
print('Duplicated data count:',dataset.duplicated().sum())

#Delete the missing datas
dataset = dataset.dropna()

dataset.isna().sum()

#Print the top 5 dan bottom 5 of the dataset
print(dataset.head())
print(dataset.tail())

#print the summary of dataset
dataset.info

#print the number of columns and rows of the dataset
print('Num of Columns, Num of Rows:', dataset.shape)

#computes and displays summary statistics for the dataset
dataset.describe()

#Gender distribution
import matplotlib.pyplot as plt
import pandas as pd

dataset.groupby(['Gender', 'satisfaction']).size().unstack().plot(kind='bar',stacked=False)
plt.xticks(rotation = 45)
plt.show()

#Customer Type distribution
import matplotlib.pyplot as plt
import pandas as pd

dataset.groupby(['Customer Type', 'satisfaction']).size().unstack().plot(kind='bar',stacked=False)
plt.xticks(rotation = 45)
plt.show()

#Type of Travel distribution
import matplotlib.pyplot as plt
import pandas as pd

dataset.groupby(['Type of Travel', 'satisfaction']).size().unstack().plot(kind='bar',stacked=False)
plt.xticks(rotation = 45)
plt.show()

#Class distribution
import matplotlib.pyplot as plt
import pandas as pd

dataset.groupby(['Class', 'satisfaction']).size().unstack().plot(kind='bar',stacked=False)
plt.xticks(rotation = 45)
plt.show()

pip install -U "scikit-learn==0.23.1"

#Separate the target and features column
target = dataset[['satisfaction']]
features = dataset.drop('satisfaction', axis=1)

# change data to categorical = binary-class
# satisfied -> 1, dissatisfied -> 0

dataset['satisfaction'] = dataset['satisfaction'].apply(lambda x: 0 if 'dissatisfied' in x else 1)

#converts categorical data of features dataset into dummy or indicator variables
features = pd.get_dummies(features)

# split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.3, random_state = 42)
print('length of X_train :', len(X_train))
print('length of y_train :', len(y_train))
print('length of X_test', len(X_test))
print('length of y_test', len(y_test))

#Satisfaction distribution
print(dataset.groupby(['satisfaction']).size())

import matplotlib.pyplot as plt
plt.figure()

print("Value Counts:\n",y_test.value_counts())
y_test.value_counts().plot.pie()
plt.title('Satisfaction', size=14)
plt.tight_layout()
plt.show()

# Import the model we are using
from sklearn.ensemble import RandomForestClassifier

# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X_train, y_train);

# Predict y data with classifier: 
y_predict = rf.predict(X_test)

# Print results: 
from sklearn.metrics import classification_report, confusion_matrix

print("confusion matrix:\n",confusion_matrix(y_test, y_predict))
print("\nModel report:\n",classification_report(y_test, y_predict))

#Accuracy
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_predict))

#Comparing the Actual class and the Predict class 
DataFinal=pd.DataFrame(X_test)
DataFinal['Actual_class']=y_test
DataFinal['Predict_class']=y_predict
print(DataFinal.head())

#Export the dataset
DataFinal.to_csv("Data Final.csv", index=False)