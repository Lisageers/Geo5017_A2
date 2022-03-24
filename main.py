import classification

path = 'data/pointclouds'

# conduct feature preparation
print('Start preparing features')
classification.feature_preparation(data_path=path)

# load the data
print('Start loading data from the local file')
ID, X, Y = classification.data_loading()
# X=features & Y=labels

# # visualize features
# print('Visualize the features')
# classification.feature_vis_multiple(X=X)

# feature_names = ['height', 'root density', 'linearity', 'planarity', 'sphericity', 'omnivariance', 'anisotropy',
#                      'eigenentropy', 'sum of the eigen-features', 'change of curvature']

# SVM classification
print('Get training set')
# omnivariance - change of curvature
set1 = [5, 9]
# height - omnivariance
set2 = [0, 5]
# height - change of curvature
set3 = [0, 9]
# 4 features
set4 = [0, 1, 5, 9]
# select from X
X_ = X[:, set2]

Xt, Yt, Xe, Ye = classification.training_set(X_, Y, 0.6)  # beware this is random
# Xt&Yt are training, Xe&Ye are evaluating/test set

# SVM classification
print('Start SVM classification')
classification.SVM_parameter_test(Xt, Yt, Xe, Ye, "linear")
classification.SVM_parameter_test(Xt, Yt, Xe, Ye, "sigmoid")
classification.SVM_parameter_test(Xt, Yt, Xe, Ye, "rbf")
classification.SVM_parameter_test(Xt, Yt, Xe, Ye, "poly")
# classification.SVM_kernel_test(Xt, Yt, Xe, Ye)
Y_svm_pred = classification.SVM_classification(Xt, Yt, Xe)

# RF classification
print('Start RF classification')
# classification.RF_parameter_test(Xt, Yt, Xe, Ye)
Y_rf_pred = classification.RF_classification(Xt, Yt, Xe)

# Evaluate results
print('Start evaluating the result')
classification.Evaluation(Y_svm_pred, Ye)
classification.Evaluation(Y_rf_pred, Ye)

# Learning curve
# classification.learningcurve(X_, Y, 'rf')