# # Task1
# ### Decision Tree
# #### Titanic data set

# importing libraries
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')


#reading the csv
titanic = pd.read_csv('data/titanic.csv')

# ### We have to clean the data now. We will drop the useless columns and fill the NA values in others

#complete missing age with median
titanic['Age'].fillna(titanic['Age'].median(), inplace = True)

#complete embarked with mode
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace = True)

#complete missing fare with median
titanic['Fare'].fillna(titanic['Fare'].median(), inplace = True)
    
#delete the cabin feature/column and others useles features.
drop_column = ['PassengerId','Cabin', 'Ticket']
titanic.drop(drop_column, axis=1, inplace = True)
titanic['FamilySize'] = titanic ['SibSp'] + titanic['Parch'] + 1
titanic['IsAlone'] = 1 #initialize to yes/1 is alone
titanic['IsAlone'].loc[titanic['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1
#quick code split title from name
titanic['title'] = titanic['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
titanic['title'] = titanic['title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
titanic['title'] = titanic['title'].replace('Mlle', 'Miss')
titanic['title'] = titanic['title'].replace('Ms', 'Miss')
titanic['title'] = titanic['title'].replace('Mme', 'Mrs')
# Now we can drop the Name column
titanic.drop('Name', axis=1, inplace = True)

# ### Now we should encode our categorical columns


titanic['title'] = titanic['title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}).fillna(0)
titanic['Sex'] = titanic['Sex'].map({'male': 1, 'female': 0})
titanic['Embarked'] = titanic['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# ### Now we can switch to model training

#Spliting the data 
X_train, X_tets, y_train, y_test = train_test_split(titanic.drop('Survived',axis=1), titanic['Survived'], test_size=0.4,random_state=41)


#Initializing our tree with deafult parameters
model = DecisionTreeClassifier(random_state=42)

model.fit(X_train,y_train)


#Getting a prediction
prediction_on_test = model.predict(X_tets)
prediction_on_train = model.predict(X_train)

#Getting accuracy
print('Task1 accuracy on test: ',accuracy_score(y_test,prediction_on_test))
print('Task1 accuracy on train: ',accuracy_score(y_train,prediction_on_train))


# Plot the accuracies versus the depth of the tree.
max_depth= range(1,20)
scores_on_test = []
scores_on_train = []
for m in max_depth:
    tree = DecisionTreeClassifier(max_depth=m,random_state=42)
    
    tree.fit(X_train,y_train)
    #Getting a prediction
    prediction_on_test = tree.predict(X_tets)
    prediction_on_train = tree.predict(X_train)
    scores_on_test.append(accuracy_score(y_test,prediction_on_test))
    scores_on_train.append(accuracy_score(y_train,prediction_on_train))

#Plotting
fig = plt.figure(figsize=(7,5))
ax = plt.subplot(111)
ax.plot(scores_on_test,label='scores_on_test')
ax.plot(scores_on_train,label='scores_on_train')
ax.set_xticks(list(range(20)))
plt.title('accuracies versus the depth of the tree')
ax.legend()
plt.show()


# ### Looking at the graph we can clearly see that the optimum value of depth for our task is 2. 
# Also, we can see that when we have maxdepth of 8, accuracy on the test set is much below 80% but on the train set it is above 95%. 
# It is a classical example of overfitting.
# If we start tunning our model's parameters using only test set, we are risking to overfit to that set too.
# To combat that we should use as much testing data as possible. There is a method called k-fold cross-validation just for that.

# In k-fold cross-validation, the model is trained $K$ times on different ($K-1$) subsets of the original dataset (in white) and checked on the remaining subset (each time a different one, shown above in orange).
# We obtain $K$ model quality assessments that are usually averaged to give an overall average quality of classification/regression.
# 
# Cross-validation provides a better assessment of the model quality on new data compared to the hold-out set approach. However, cross-validation is computationally expensive when you have a lot of data.
#  
# Cross-validation is a very important technique in machine learning and can also be applied in statistics and econometrics. It helps with hyperparameter tuning, model comparison, feature evaluation, etc.

# ### Now we can actually perform cross-validation.


model = DecisionTreeClassifier(max_depth=2)
mean_fold_score=cross_val_score(model,titanic.drop('Survived',axis=1), titanic['Survived'] ,cv = 5, scoring = "accuracy")
print('Task1 max_depth=2 cross-validated accuracy: ',np.mean(mean_fold_score))


# ### It is obvious that this result is much more trustful, thus we will do our accuracies versus the depth of the tree graph again

# Plot the accuracies versus the depth of the tree.
max_depth= range(20)
cv_scores = [ ]
for m in max_depth:
    tree = DecisionTreeClassifier(max_depth=m,random_state=42)
    scores = cross_val_score(tree,titanic.drop('Survived',axis=1), titanic['Survived'] ,cv = 5, scoring = "accuracy")
    cv_scores.append(scores.mean())
#Plotting
fig = plt.figure(figsize=(7,5))
ax = plt.subplot(111)
ax.plot(cv_scores,label='scores_on_test')
ax.set_xticks(list(range(20)))
plt.title('accuracies versus the depth of the tree')
ax.legend()
plt.show()


# ### After cross-validation we can see that actually, the best depth of the tree is 4

model = DecisionTreeClassifier(max_depth=4)
mean_fold_score=cross_val_score(model,titanic.drop('Survived',axis=1), titanic['Survived'] ,cv = 5, scoring = "accuracy")
print('Task1 max_depth=4 cross-validated accuracy: ',np.mean(mean_fold_score))


# # Task2
# ### Support Vector Machine
# #### Wisconsin Breast Cancer data set

#reading the csv
BreastCancer = pd.read_csv('data/breastcancerwisconsin.txt',names=['id','Clump Thickness','Uniformity of Cell Size',
                                                              'Uniformity of Cell Shape','Marginal Adhesion',
                                                              'Single Epithelial Cell Size','Bare Nuclei',
                                                              'Bland Chromatin','Normal Nucleoli', 'Mitoses', 'Class'])
BreastCancer=BreastCancer.replace('?',np.nan)
BreastCancer.drop('id',axis=1,inplace=True)



# We have 16 missing values in Bare Nuclei column


BreastCancer['Bare Nuclei'].fillna(BreastCancer['Bare Nuclei'].mode()[0], inplace = True)
BreastCancer['Bare Nuclei']=BreastCancer['Bare Nuclei'].astype('int64')





#lets encode our target deature in a more standart way
BreastCancer['Class'] = BreastCancer['Class'].map({2: 0, 4: 1})
BreastCancer['Class'].value_counts()




# performing data visualization with scatter plots
sns.pairplot(BreastCancer,corner=True,diag_kind='hist',hue = 'Class')



#Splitiing the data
X_train, X_tets, y_train, y_test = train_test_split(BreastCancer.drop('Class',axis=1), BreastCancer['Class'], test_size=0.4,random_state=41)


#Lets Train our model
model = SVC()
model.fit(X_train,y_train)
prediction_on_test = model.predict(X_tets)
prediction_on_train = model.predict(X_train)
print('Task2 accurasy on test: ',accuracy_score(y_test,prediction_on_test))
print('Task2 accurasy on train: ',accuracy_score(y_train,prediction_on_train))


# ### We can see that our model performs really well (accuracy 97%+) even with the default parameters

# ### Let's check it using cross-validation now


model =SVC()
mean_fold_score=cross_val_score(model,BreastCancer.drop('Class',axis=1), BreastCancer['Class'],cv = 5, scoring = "accuracy")
print('Task2 cross-validated accuracy: ',np.mean(mean_fold_score))


# ### Now we can plot the decision region of the classifier


# Now we will reduce dimentionality into two-dimensional feature space using PCA. By reducing from 11 into 2 dimensions we lost only 36% of varience
pca = PCA(0.76)
y_reduced=BreastCancer['Class']
X_reduced = pca.fit_transform(BreastCancer.drop('Class',axis=1))


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy



def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


model = SVC()
model.fit(X_reduced,y_reduced)
plt.figure(figsize=(10,7))
ax = plt.subplot(111)
X0, X1 = X_reduced[:, 0], X_reduced[:, 1]
xx, yy = make_meshgrid(X0, X1)
plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=BreastCancer['Class'], cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('SVC decision region')
plt.show()
print('number of support vectors to each class', model.n_support_)