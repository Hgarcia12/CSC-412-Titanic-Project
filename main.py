import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error

#reads from data sample of Titanic survivors
df = pd.read_csv('/Users/hagigarcia/PycharmProjects/TitanicProject/data/train.csv',header=0)

"""
Checking for missing data
print(df.isna().any().any())#checks for missing data
df.loc[:, df.isnull().any()].columns
df_no_na = df.dropna()
"""

print("Data Statistics: \n",df.describe())#prints statistics of the data

#Data Cleaning
df = df.drop(['PassengerId','Ticket','Name'],axis=1)#Data sets not needed, too many variables

df['Sex'].replace({'female':0,'male':1},inplace=True)#gives sex a binary value
df['Sex'].unique()

df['Cabin'] = df['Cabin'].replace(np.nan,0)#replaces missing values for cabin with 0
df['Cabin']=df['Cabin'].apply(lambda x: x if np.isreal(x) else 1)#if they have a cabin its 1
df['Cabin'].unique()

#splits fare into groups to reduce useless data
df = df.astype({"Fare": int})
df.loc[df['Fare'] < 8, 'Fare'] = 0
df.loc[(df['Fare'] >= 8) & (df['Fare'] < 15), 'Fare'] = 1
df.loc[(df['Fare'] >= 15) & (df['Fare'] < 31), 'Fare'] = 2
df.loc[(df['Fare'] >= 31) & (df['Fare'] < 99), 'Fare'] = 3
df.loc[(df['Fare'] >= 99) & (df['Fare'] < 250), 'Fare'] = 4
df.loc[df['Fare'] >= 250, 'Fare'] = 5

#df['Embarked'].describe() shows S to be most popular spot
#so we will use it to fill missing values
pop_val = "S"
df['Embarked'] = df['Embarked'].fillna(pop_val)
df['Embarked'] = df['Embarked'].replace({"S":0,"C":1,"Q":2})#making the embark points numerical
#print("missing: ",df['Embarked'].isna().any().any()) #to check for missing values in embark
df['Embarked'] = df['Embarked'].apply(np.int64)#converts Embarked to int64

#cleaning up age by generating random age from dataset we have for missing values
mean_age = df['Age'].mean()
std_age = df['Age'].std()
null_age = df['Age'].isnull().sum()#number of missing values for age
rand_age = np.random.randint(mean_age - std_age, mean_age + std_age, size = null_age)#generating random age based on our data
no_null_age = df['Age'].copy()
no_null_age[np.isnan(no_null_age)] = rand_age
df['Age'] = no_null_age
df["Age"] = df["Age"].astype(int)#converts Age to int64 type

#splitting age into groups to make it easier
df.loc[df['Age'] < 14, 'Age'] = 0
df.loc[(df['Age'] >= 14) & (df['Age'] < 20), 'Age'] = 1
df.loc[(df['Age'] >= 20) & (df['Age'] < 25), 'Age'] = 2
df.loc[(df['Age'] >= 25) & (df['Age'] < 30), 'Age'] = 3
df.loc[(df['Age'] >= 30) & (df['Age'] < 36), 'Age'] = 4
df.loc[(df['Age'] >= 36) & (df['Age'] < 44), 'Age'] = 5
df.loc[(df['Age'] >= 44), 'Age'] = 6

#Generates set to train and test from df
X = df.drop('Survived',axis=1)#features
y = df['Survived']#to be predicted
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5,random_state=42)#uses x and y df and half the set

#Logistical Regression
print("---Logistic Regression Data---")
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
y_pred = log_reg.predict(X_test)
logreg_acc = round(log_reg.score(X_train,y_train) * 100, 2)#accuracy score percent
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
#prints training and test scores
print('Training Score: {}'.format(log_reg.score(X_train,y_train)))
print('Test Score: {}'.format(log_reg.score(X_test,y_test)))
print('RMSE: {}'.format(rmse))

#Ridge regularization
steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Ridge(alpha=10, fit_intercept=True))
]
ridge_pipe = Pipeline(steps)
ridge_pipe.fit(X_train, y_train)

print('\nTraining Score Ridge Regression: {}'.format(ridge_pipe.score(X_train, y_train)))
print('Test Score Ridge Regression: {}'.format(ridge_pipe.score(X_test, y_test)))

#confusion matrix heat map for logistic regression
plt.figure(figsize=(5,5))
ConfMatrix = confusion_matrix(y_test,log_reg.predict(X_test))
sns.heatmap(ConfMatrix,annot=True, cmap="Reds", fmt="d",xticklabels = ['Survived', 'Died'],yticklabels = ['Survived', 'Died'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Survival - Confusion Matrix");
plt.show()

#Decision Tree
decision_tree = DecisionTreeClassifier(max_depth=11)
decision_tree.fit(X_train, y_train)#fits to the training sets
dtree_pred = decision_tree.predict(X_test)
dtree_acc = round(decision_tree.score(X_train, y_train) * 100, 2)#Accuracy rounded
dtree_text = tree.export_text(decision_tree)
print("---Decision Tree Text Repesentation---")
print(dtree_text)#prints text version of decision tree

#Perceptron
perc_sur = Perceptron()
perc_sur.fit(X_train,y_train)
per_pred = perc_sur.predict(X_test)#uses test data
perc_acc = round(perc_sur.score(X_train,y_train) * 100,2)

#Scores
print("----Model Scores----")
print("Logisstical Regression: ",logreg_acc)
print("Decision Tree: ",dtree_acc)
print("Perceptron Score: ",perc_acc)

print("----Cross Validation Scores----")
log_cross_score = cross_val_score(log_reg,X_train,y_train,cv=10,scoring="accuracy")
dtree_cross_score = cross_val_score(decision_tree,X_train,y_train,cv=10,scoring="accuracy")
perc_cross_score = cross_val_score(perc_sur,X_train,y_train,cv=10,scoring="accuracy")

print("Logistical Regression Score: ",log_cross_score)
print("Mean: ",log_cross_score.mean())
print("Standard Deviation: ",log_cross_score.std())
print("Decision Tree Score: ",dtree_cross_score)
print("Mean: ",dtree_cross_score.mean())
print("Standard Deviation: ",dtree_cross_score.std())
print("Perceptron Score: ",perc_cross_score)
print("Mean: ",perc_cross_score.mean())
print("Standard Deviation: ",perc_cross_score.std())

print("---ROC AUC---")
#below is to test for accuracy for decision tree

y_scores = decision_tree.predict_proba(X_train)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(y_train, y_scores)
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_scores)
# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.title("ROC AUC Decision Tree")
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()
#ROC AUC for highest scoring model
r_a_score = roc_auc_score(y_train, y_scores)
print("ROC-AUC-Score for Decision Tree: ", r_a_score)