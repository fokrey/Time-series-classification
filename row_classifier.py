import numpy as np 
import random 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline


T = np.arange(0, 10000, 1)
#print(T)

def func0(x):
    random.seed(874390)
    return 1.5*np.sin(2*x) + 1.99 + np.random.normal(0, 1, x.shape[0])

def func1(x):
    random.seed(19700)
    return 2*np.sin(2.5*x) + 1.7 + np.random.normal(0, 1, x.shape[0])

def func2(x):
    random.seed(1580)
    return 1.6*np.sin(3*x) + 1.9 + np.random.normal(0, 1, x.shape[0])

def func3(x):
    random.seed(15640)
    return 3*np.sin(2.1*x) + 1.5 + np.random.normal(0, 1, x.shape[0])

def func4(x):
    random.seed(1032452)
    return 2.5*np.sin(2.6*x) + 1 + np.random.normal(0, 1, x.shape[0])

N_classes = 5
N_train_for_each  = 50
N_test_for_each = 50
N_features = 2

def calc_features (x):
    # x is numpy array of length 1000
    features = np.zeros (N_features)
    features[0] = np.average (x)
    features[1] = np.max (x) - np.min (x)
    return features

X = []
X.append(func0(T))
X.append(func1(T))
X.append(func2(T))
X.append(func3(T))
X.append(func4(T))

y_train = np.zeros(N_classes * N_train_for_each)
y_test = np.zeros(N_classes * N_test_for_each)
x_train = np.zeros((N_classes * N_train_for_each, N_features))
x_test = np.zeros((N_classes * N_test_for_each, N_features))

for num_class in range(0, N_classes):
    for num_sample in range(0, N_train_for_each):
        start = int(random.random () * 9000)
        y_train[num_class * N_train_for_each + num_sample] = num_class
        x_train[num_class * N_train_for_each + num_sample] = calc_features (X[num_class][start:start+1000])
        
for num_class in range(0, N_classes):
    for num_sample in range(0, N_test_for_each):
        start = int(random.random () * 9000)
        y_test[num_class * N_test_for_each + num_sample] = num_class
        x_test[num_class * N_test_for_each + num_sample] = calc_features (X[num_class][start:start+1000])
        
# fit classifier

pipeline = Pipeline([("scaler", StandardScaler()),
                       ("basic_logreg", LogisticRegression(multi_class='multinomial',solver='saga',tol=1e-3,max_iter=500))])
param_grid = {
    'basic_logreg__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'basic_logreg__penalty': ['l1', 'l2'],
}

grid_pipeline = GridSearchCV(pipeline,param_grid)
grid_pipeline.fit(x_train,y_train)

print(grid_pipeline.best_params_)

y_pred = grid_pipeline.best_estimator_.predict(x_test)

print(accuracy_score(y_test, y_pred))
#print(f1_score(y_train, y_pred, average=None))
        

#scipy.signal.find_peaks()

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(cm)
ax.grid(False)
ax.set_xlabel('Predicted outputs', color='black')
ax.set_ylabel('Actual outputs', color='black')
ax.xaxis.set(ticks=range(N_classes))
ax.yaxis.set(ticks=range(N_classes))
ax.set_ylim(4.5, -0.5)
for i in range(N_classes):
    for j in range(N_classes):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
plt.show()


h = 0.02  # step size in the mesh

alphas = np.logspace(-1, 1, 1)

classifiers = []
names = []
for alpha in alphas:
    classifiers.append(
        make_pipeline(
            StandardScaler(),
            MLPClassifier(
                solver="lbfgs",
                alpha=alpha,
                random_state=1,
                max_iter=2000,
                early_stopping=True,
                hidden_layer_sizes=[10, 10],
            ),
        )
    )
    names.append(f"alpha {alpha:.2f}")

X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=0, n_clusters_per_class=1
)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [
    make_moons(noise=0.3, random_state=0),
    make_circles(noise=0.2, factor=0.5, random_state=1),
    linearly_separable,
]

figure = plt.figure(figsize=(17, 9))
i = 1
# iterate over datasets
for X, y in datasets:
    # split into training and test part

    x_min, x_max = x_train[:, 0].min() - 0.5, x_train[:, 0].max() + 0.5
    y_min, y_max = x_train[:, 1].min() - 0.5, x_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF", "#FFF700", "#00FF51", "#9900FF"])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    # Plot the training points
    ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max] x [y_min, y_max].

        # Plot also the training points
        ax.scatter(
            x_train[:, 0],
            x_train[:, 1],
            c=y_train,
            cmap=cm_bright,
            edgecolors="black",
            s=25,
        )
        # and testing points
        ax.scatter(
            x_test[:, 0],
            x_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            alpha=0.6,
            edgecolors="black",
            s=25,
        )

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        i += 1

figure.subplots_adjust(left=0.02, right=0.98)
plt.show()