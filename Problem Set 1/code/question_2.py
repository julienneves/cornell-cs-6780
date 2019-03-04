import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import make_pipeline

# Separate dataset into training/test set
df = pd.read_csv("code/diabetic_retinopathy.csv", header=None)

X = df.iloc[:, :-1]
y = df.iloc[:,-1]
X_train, X_test = np.split(X, [900], axis = 0)
y_train, y_test = np.split(y, [900], axis = 0)

X_train.shape


# Part (a)
print("Part (a)")
print("Training set - Label count")
print(y_train.value_counts())

majority_label = y_train.mode()[0] # Find majority label
y_test_pred = pd.Series(majority_label, index=np.arange(len(y_test)))
y_train_pred = pd.Series(majority_label, index=np.arange(len(y_train)))

print("Accuracy score (train): ",accuracy_score(y_train, y_train_pred))
print("Accuracy score (test): ",accuracy_score(y_test, y_test_pred))

# Part (b)
print("Part (b)")

classifier = DecisionTreeClassifier(random_state = 347)
classifier.fit(X_train, y_train)

y_test_pred = classifier.predict(X_test)
y_train_pred = classifier.predict(X_train)

print("Accuracy score (train): ",accuracy_score(y_train, y_train_pred))
print("Accuracy score (test): ",accuracy_score(y_test, y_test_pred))

# Part (c)
print("Part (c)")
print("Max depth: ", classifier.tree_.max_depth)

max_depth = range(1,11)

val_score = []
train_score = []

for i in max_depth:
    classifier = DecisionTreeClassifier(max_depth = i, random_state = 347).fit(X_train, y_train)
    y_test_pred, y_train_pred = classifier.predict(X_test), classifier.predict(X_train)
    train_score.append(accuracy_score(y_train, y_train_pred))
    val_score.append(accuracy_score(y_test, y_test_pred))

fig1 = plt.figure()
ax1 = plt.axes()
ax1.plot(max_depth, train_score, color ='blue', label = 'training score')
ax1.plot(max_depth, val_score, color ='red', label = 'prediction score')
ax1.set_xlabel('max depth')
ax1.set_ylabel('score')
ax1.legend(loc='best');

fig1.savefig('fig_1.png')

plt.plot(max_depth, train_score, color ='blue', label = 'training score')
plt.plot(max_depth, val_score, color ='red', label = 'training score')

# Part (d)
print("Part (d)")

# Part (e)
print("Part (e)")
train_score, val_score = validation_curve(classifier, X_train, y_train, 'max_depth', max_depth, cv = 4)

fig2 = plt.figure()
ax2 = plt.axes()
ax2.plot(max_depth, np.mean(train_score,1), color = 'blue', label = 'training score')
ax2.plot(max_depth, np.mean(val_score,1), color = 'red', label = 'validation score')
ax2.legend(loc='best')
ax2.set_xlabel('max depth')
ax2.set_ylabel('score');
fig2.savefig('fig_2.png')

max_depth
val_score
np.mean(val_score,1)

max_depth_best = max_depth[np.argmax(np.mean(val_score,1))]
print("Max depth with best performance: ", max_depth_best)

classifier = DecisionTreeClassifier(max_depth = max_depth_best, random_state = 347).fit(X_train, y_train)
y_test_pred, y_train_pred = classifier.predict(X_test), classifier.predict(X_train)
print("Accuracy score (train): ",accuracy_score(y_train, y_train_pred))
print("Accuracy score (test): ",accuracy_score(y_test, y_test_pred))


# Part (f)
print("Part (f)")

model = make_pipeline(SelectKBest(chi2),
                      DecisionTreeClassifier(max_depth = 10, random_state = 347))
features = range(1,11)

train_score, val_score = validation_curve(model, X_train, y_train, param_name = "selectkbest__k", param_range = features, cv = 4)
fig3 = plt.figure()
ax3 = plt.axes()
ax3.plot(features, np.mean(train_score,1), color = 'blue', label = 'training score')
ax3.plot(features, np.mean(val_score,1), color = 'red', label = 'validation score')
ax3.legend(loc='best')
ax3.set_xlabel('features')
ax3.set_ylabel('score');
fig3.savefig('fig_3.png')


features_best = features[np.argmax(np.mean(val_score,1))]
print("Number features with best performance: ", features_best)

model.set_params(selectkbest__k = features_best)
model.fit(X_train, y_train)
y_test_pred, y_train_pred = model.predict(X_test), model.predict(X_train)
print("Accuracy score (train): ",accuracy_score(y_train, y_train_pred))
print("Accuracy score (test): ",accuracy_score(y_test, y_test_pred))
