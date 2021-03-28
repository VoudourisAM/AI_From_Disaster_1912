# Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#Load the data
Train_D = pd.read_csv('.../Project_Titanic/Data/train.csv')
Test_D = pd.read_csv('.../Project_Titanic/Data/test.csv')

Real_D = pd.read_csv('.../Project_Titanic/Data/gender_submission.csv')
Real_D.drop(['PassengerId'], axis=1, inplace=True)

Train_D

Train_D.describe()

print(Train_D.isnull().sum())
print()
print(Test_D.isnull().sum())

#---Diagrams---#
sns.heatmap(Train_D.isnull())

Train_D['Survived'].value_counts().plot.pie(autopct='%1.2f%%')
plt.xlabel('0=Dead   -   1=Survived')
plt.show()

sns.countplot(x='Survived' , data=Train_D, hue='Sex')
plt.xlabel('Dead            -            Survived')
plt.ylabel('Count')
plt.show()

sns.countplot(x='Embarked', data=Train_D, hue='Survived')
plt.xlabel('Dead            -            Survived')
plt.ylabel('0=Dead   -   1=Survived')
plt.show()

sns.countplot(x='Pclass', data=Train_D, hue='Sex')

pd.pivot_table(Train_D, index='Pclass', values='Age')

sns.countplot(x='Survived', data=Train_D, hue='Pclass')
plt.xlabel('Dead            -            Survived')
plt.ylabel('Count')
plt.show()
#---Diagrams---#

#Drop columns Train,Test 
Train_D = Train_D.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1)
Test_D = Test_D.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1 )

#Cleaning data
Train_D['Embarked'] = Train_D['Embarked'].fillna('S')

sns.displot(Train_D['Age'])

Train_D['Age'] = Train_D['Age'].fillna(Train_D['Age'].median())

Test_D['Age'] = Test_D['Age'].fillna(Train_D['Age'].median())
Test_D['Fare'] = Test_D['Fare'].fillna(Train_D['Fare'].median())

print(Train_D.isnull().sum())
print()
print(Test_D.isnull().sum())

print(Train_D.dtypes)
print()
print(Test_D.dtypes)

#Encode the categorical data#
Train_D.iloc[:, 2] = Train_D.Name.str.extract( '([A-Za-z]+)\.', expand=False )
Test_D.iloc[:, 1] = Test_D.Name.str.extract( '([A-Za-z]+)\.', expand=False )
print(Train_D['Name'].value_counts())
print()
print(Test_D['Name'].value_counts())

Title = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Dr':5, 'Rev':5,
        'Mlle':5, 'Major':5, 'Col':5, 'Mme':5, 'Sir':5, 'Lady':5,
        'Ms':5, 'Countess':5, 'Capt':5, 'Jonkheer':5, 'Don':5, 'Dona':5}

Train_D['Name'] = Train_D['Name'].map(Title)
Test_D['Name'] = Test_D['Name'].map(Title)

print(Train_D['Embarked'].unique())

Towns = {'S':1, 'C':2, 'Q':3}
Train_D['Embarked'] = Train_D['Embarked'].map(Towns)
Test_D['Embarked'] = Test_D['Embarked'].map(Towns)

print(Train_D['Embarked'].unique())

print(Train_D.dtypes)
print()
print(Test_D.dtypes)

Train_D.head(5)

#Splitting the data columns
X = Train_D.iloc[:, 1:10].values
Y = Train_D.iloc[:, 0].values

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=0)

#Scale the Train,Test data
from sklearn.preprocessing import StandardScaler
Sc = StandardScaler()

X_Train = Sc.fit_transform(X_Train)
X_Test = Sc.fit_transform(X_Test)

#Create the machine-learning models
def models(X_Train, Y_Train):
    

    #Logistic-Regrassion
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state = 0)
    log.fit(X_Train, Y_Train)
    print('Logistic-Regression Train Accuracy: ', log.score(X_Train, Y_Train))
# ------------------------------------------------------------------------------------------
    #Decision-Tree
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    tree.fit(X_Train, Y_Train)
    print('Decision-Tree Train Accuracy: ', tree.score(X_Train, Y_Train))
#-------------------------------------------------------------------------------------------

    #Random-Forest_Classifier
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=20,
                                    criterion='entropy',
                                    min_samples_split=7,
                                    random_state=5,
                                    bootstrap=True)
    forest.fit(X_Train, Y_Train)
    print('Random-Forest_Classifier Train Accuracy: ', forest.score(X_Train, Y_Train))
    
#--------------------------------------------------------------------------------------------    

    return log, tree, forest

#Print the Training accuracy
Model = models(X_Train, Y_Train)

#Print t he Testing accuracy
from sklearn.metrics import confusion_matrix

for i in range(len(Model)):
    Acc_ts_Model = confusion_matrix(Y_Test, Model[i].predict(X_Test))
    
    TN, FP, FN, TP = confusion_matrix(Y_Test, Model[i].predict(X_Test)).ravel()
    
    Test_Score = (TP + TN) / (TP + TN + FN + FP)
    
    print(Acc_ts_Model)
    print('Model [{}] Testing Accuracy = "{}" '.format(i, Test_Score))
    print('---------------------------------------------------------')


#Print the prediction of the Logistic-Regression
Pred_Lr = Model[0].predict(X_Test)
print('Logistic-Regression Model:')
print(Pred_Lr)

print()
print('------------------------------------------------------------------')
print()

#Print the prediction of the Decision-Tree
Pred_Tree = Model[1].predict(X_Test)
print('Decision-Tree Model:')
print(Pred_Tree)

print()
print('------------------------------------------------------------------')
print()

#Print the prediction of the Random-Forest-Classifier
Pred_Rf = Model[2].predict(X_Test)
print('Random-Forest Model:')
print(Pred_Rf)


print()
print('------------------------------------------------------------------')
print()

#Print the actual values 
print('Real-Data')
print(Y_Test)

#Prediction on Test data
Test_D.head(6)

Rf_PredT = Model[2].predict(Test_D)
Test_D['Real_Data_Surv'] = pd.DataFrame(Real_D['Survived'])
Test_D['My_Pred_Rf'] = Rf_PredT
Test_D.iloc[:16 , :]

#How accurate was the Random-Forest model at the current Test-Data
Count0 = 0
Count1 = 1
Count2 = 0

for i in range(len(Test_D)):
    Num_RD = Test_D.iloc[i,8]
    Num_MD = Test_D.iloc[i,9]
    if Num_RD == 0 and Num_MD == 0:
        Count0 += 1
    elif Num_RD == 1 and Num_MD == 1:
        Count1 += 1
    else:
        Count2 += 1
        
print('The rate of passengers that did not survive : ',Count0)
print('The rate of passengers that did survive : ',Count1)
print()
print('The Random-Forest-Model from 418 rows, was wrong {} rows'.format(Count2-1))
print('Random-Forest-Model is {:.2f}% right'.format(Count2/418))