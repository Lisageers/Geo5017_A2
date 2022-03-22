import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm
from os.path import exists, join


class urban_object:
    """
    Define an urban object
    """
    def __init__(self, filenm):
        """
        Initialize the object
        """
        # obtain the cloud name
        self.cloud_name = filenm.split('/\\')[-1][-7:-4]

        # obtain the cloud ID
        self.cloud_ID = int(self.cloud_name)

        # obtain the label
        self.label = math.floor(1.0*self.cloud_ID/100)

        # obtain the points
        self.points = read_xyz(filenm)

        # initialize the feature vector
        self.feature = []

    def compute_features(self):
        """
        Compute the features, here we provide two example features. You're encouraged to design your own features
        """
        # calculate the height
        height = np.amax(self.points[:, 2])
        self.feature.append(height)

        # get the root point
        root = self.points[[np.argmin(self.points[:, 2])]]

        # construct the 2D kd tree
        kd_tree_2d = KDTree(self.points[:, :2], leaf_size=5)

        # compute the root point planar density
        radius_root = 0.2
        count = kd_tree_2d.query_radius(root[:, :2], r=radius_root, count_only=True)
        root_density = 1.0*count[0] / len(self.points)
        self.feature.append(root_density)


def read_xyz(filenm):
    """
    Reading points
        filenm: the file name
    """
    points = []
    with open(filenm, 'r') as f_input:
        for line in f_input:
            p = line.split()
            p = [float(i) for i in p]
            points.append(p)
    points = np.array(points).astype(np.float32)
    return points


def feature_preparation(data_path):
    """
    Prepare features of the input point cloud objects
        data_path: the path to read data
    """
    # check if the current data file exist
    data_file = 'data.txt'
    if exists(data_file):
        return

    # obtain the files in the folder
    files = np.sort(glob.glob(join(data_path, '*.xyz')))

    # initialize the data
    input_data = []

    # get the overall point cloud file number
    N = len(files)

    # initialize the progress bar
    progress = tqdm(range(N))

    # retrieve each data object and obtain the feature vector
    for i in progress:
        # get the file
        file_name = files[i]

        # read data
        i_object = urban_object(filenm=file_name)

        # calculate features
        i_object.compute_features()

        # add the data to the list
        i_data = [i_object.cloud_ID, i_object.label] + i_object.feature
        input_data += [i_data]

        # set progress description
        progress.set_description("Processed files: ")

    # transform the output data
    outputs = np.array(input_data).astype(np.float32)

    # write the output to a local file
    data_header = 'ID,label,height,root_density'
    np.savetxt(data_file, outputs, fmt='%10.5f', delimiter=',', newline='\n', header=data_header)


def data_loading(data_file='data.txt'):
    """
    Read the data with features from the data file
        data_file: the local file to read data with features and labels
    """
    # load data
    data = np.loadtxt(data_file, dtype=np.float32, delimiter=',', comments='#')

    # extract object ID, feature X and label Y
    ID = data[:, 0].astype(np.int32)
    Y = data[:, 1].astype(np.int32)
    X = data[:, 2:].astype(np.float32)

    return ID, X, Y


def feature_visualization(X):
    """
    Visualize the features
        X: input features. This assumes classes are stored in a sequential manner
    """
    # initialize a plot
    fig, ax = plt.subplots()
    plt.title("feature subset visualization of 5 classes", fontsize="small")

    # define the labels and corresponding colors
    colors = ['firebrick', 'grey', 'darkorange', 'dodgerblue', 'olivedrab']
    labels = ['building', 'car', 'fence', 'pole', 'tree']

    # plot the data with first two features
    for i in range(5):
        ax.scatter(X[100*i:100*(i+1), 0], X[100*i:100*(i+1), 1], marker="o", c=colors[i], edgecolor="k", label=labels[i])

    # show the figure with labels
    """
    Replace the axis labels with your own feature names
    """
    plt.xlabel('x1: height')
    plt.ylabel('x2: root density')
    ax.legend()
    plt.show()

def training_set(X, Y, t):
    """
    Conduct SVM classification
        X: features
        Y: labels
        t: train size, between 0 and 1
    """
    Xt, Xe, Yt, Ye = train_test_split(X, Y, train_size=t)

    return Xt, Yt, Xe, Ye


def plotSVC(titles, X, models):
    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]  # decide which features we want to use to test the kernel methods
    x_min, x_max = X0.min() - 1, X0.max() + 1
    y_min, y_max = X1.min() - 1, X1.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    for clf, title, ax in zip(models, titles, sub.flatten()):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z)
        ax.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel("Height")  # change this after feature selection!!
        ax.set_ylabel("Root density")  # change this after feature selection!!
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    plt.show()


def SVM_parameter_test(X, Y):
    # test C of linear kernel svm
    models = (
        svm.SVC(kernel="linear", C=0.1),
        svm.SVC(kernel="linear", C=1),
        svm.SVC(kernel="linear", C=10),
        svm.SVC(kernel="linear", C=100)
    )
    models = (clf.fit(X, Y) for clf in models)
    titles = (
        "SVC with linear kernel C=0.1",
        "SVC with linear kernel C=1",
        "SVC with linear kernel C=10",
        "SVC with linear kernel C=100"
    )
    plotSVC(titles, X, models)

    # test degree of poly kernel svm
    models = (
        svm.SVC(kernel="poly", degree=0),
        svm.SVC(kernel="poly", degree=2),
        svm.SVC(kernel="poly", degree=4),
        svm.SVC(kernel="poly", degree=6)
    )
    models = (clf.fit(X, Y) for clf in models)
    titles = (
        "SVC with poly kernel degree=0",
        "SVC with poly kernel degree=2",
        "SVC with poly kernel degree=4",
        "SVC with poly kernel degree=6"
    )
    plotSVC(titles, X, models)

    # test gamma of rbf kernel svm
    models = (
        svm.SVC(kernel="rbf", gamma='auto'),
        svm.SVC(kernel="rbf", gamma='scale'),
        svm.SVC(kernel="rbf", gamma=10),
        svm.SVC(kernel="rbf", gamma=100)
    )
    models = (clf.fit(X, Y) for clf in models)
    titles = (
        "SVC with rbf kernel gamma=1 / n_features",
        "SVC with rbf kernel gamma=1 / (n_features * X.var())",
        "SVC with rbf kernel gamma=10",
        "SVC with rbf kernel gamma=100"
    )
    plotSVC(titles, X, models)

def SVM_kernel_test(X, Y):
    """
    Conduct SVM kernel testing
        X: features
        Y: labels
    """
    C = 1.0  # SVM regularization parameter
    models = (
        svm.SVC(kernel="linear", C=C),
        svm.SVC(kernel="sigmoid", C=C, gamma="auto"),
        svm.SVC(kernel="rbf", gamma=0.7, C=C),
        svm.SVC(kernel="poly", degree=3, gamma="auto", C=C),
    )
    models = (clf.fit(X, Y) for clf in models)

    # title for the plots
    titles = (
        "SVC with linear kernel",
        "SVC with sigmoid kernel",
        "SVC with RBF kernel",
        "SVC with polynomial kernel",
    )

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]  # decide which features we want to use to test the kernel methods
    x_min, x_max = X0.min() - 1, X0.max() + 1
    y_min, y_max = X1.min() - 1, X1.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    for clf, title, ax in zip(models, titles, sub.flatten()):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z)
        ax.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel("Height")  # change this after feature selection!!
        ax.set_ylabel("Root density")  # change this after feature selection!!
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    plt.show()

def SVM_classification(Xt, Yt, Xe):
    """
    Conduct SVM classification
        Xt: features training set
        Yt: labels training set
        Xe: features evaluation/test set
    """
    C = 1.0  # SVM regularization parameter
    clf = svm.SVC(kernel="linear", C=C) # need to change the kernel after tests!!
    clf.fit(Xt, Yt)
    predicted_labels = clf.predict(Xe)
    return predicted_labels


def RF_parameter_test(X, Y):
    # test nr of trees
    models = (
        RandomForestClassifier(n_estimators=1),
        RandomForestClassifier(n_estimators=10),
        RandomForestClassifier(n_estimators=100),
        RandomForestClassifier(n_estimators=1000)
    )
    models = (clf.fit(X, Y) for clf in models)
    titles = (
        "RF with 1 tree",
        "RF with 10 trees",
        "RF with 100 trees",
        "RF with 1000 trees"
    )
    plotSVC(titles, X, models)

    # test criterion
    models = (
        RandomForestClassifier(criterion="gini"),
        RandomForestClassifier(criterion="entropy")
    )
    models = (clf.fit(X, Y) for clf in models)
    titles = (
        "RF with gini",
        "RF with entropy"
    )
    plotSVC(titles, X, models)

    # test max features considered at split
    models = (
        RandomForestClassifier(max_features="auto"),
        RandomForestClassifier(max_features="log2"),
        RandomForestClassifier(max_features=1),
        RandomForestClassifier(max_features=None),
    )
    models = (clf.fit(X, Y) for clf in models)
    titles = (
        "RF with max_features=sqrt(n_features)",
        "RF with max_features=log2(n_features)",
        "RF with max_features=1",
        "RF with max_features=n_features"
    )
    plotSVC(titles, X, models)


def RF_classification(Xt, Yt, Xe):
    """
    Conduct RF classification
        Xt: features training set
        Yt: labels training set
        Xe: features evaluation/test set
    """
    clf = RandomForestClassifier(n_estimators=100, criterion="gini", max_features="auto") # we should do tests with these parameters
    clf.fit(Xt, Yt)
    predicted_labels = clf.predict(Xe)
    return predicted_labels


def Evaluation(Y_pred=None, Y_true=None):
    """
    Evaluate the performance
        Y_preds: predicted labels
        Y_true: true labels
    """
    print("Overall accuracy:")
    print(accuracy_score(Y_true, Y_pred)) # 0 to 1

    print("mean per-class accuracy")
    cmatrix = confusion_matrix(Y_true, Y_pred)

    print(cmatrix.diagonal()/cmatrix.sum(axis=1))

    print("Confusion matrix:")
    print(cmatrix)



if __name__=='__main__':
    # specify the data folder
    """"Here you need to specify your own path"""
    path = 'data/pointclouds'

    # conduct feature preparation
    print('Start preparing features')
    feature_preparation(data_path=path)

    # load the data
    print('Start loading data from the local file')
    ID, X, Y = data_loading()
    # X=features & Y=labels

    # visualize features
    # print('Visualize the features')
    # feature_visualization(X=X)

    # SVM classification
    print('Get training set')
    Xt, Yt, Xe, Ye = training_set(X, Y, 0.6) # beware this is random
    # Xt&Yt are training, Xe&Ye are evaluating/test set
    
    # SVM classification
    print('Start SVM classification')
    SVM_parameter_test(X, Y) # not sure if we should use training or whole set?
    SVM_kernel_test(X, Y)
    Y_svm_pred = SVM_classification(Xt, Yt, Xe)

    # RF classification
    print('Start RF classification')
    RF_parameter_test(X, Y)
    Y_rf_pred = RF_classification(Xt, Yt, Xe)

    # Evaluate results
    print('Start evaluating the result')
    Evaluation(Y_svm_pred, Ye)
    Evaluation(Y_rf_pred, Ye)
