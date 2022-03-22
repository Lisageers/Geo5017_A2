import classification

path = 'data/pointclouds'

# conduct feature preparation
print('Start preparing features')
classification.feature_preparation(data_path=path)

# load the data
print('Start loading data from the local file')
ID, X, Y = classification.data_loading()
# X=features & Y=labels

# visualize features
# print('Visualize the features')
# feature_visualization(X=X)

# SVM classification
print('Get training set')
Xt, Yt, Xe, Ye = classification.training_set(X, Y, 0.6)  # beware this is random
# Xt&Yt are training, Xe&Ye are evaluating/test set

# SVM classification
print('Start SVM classification')
classification.SVM_parameter_test(X, Y) # not sure if we should use training or whole set?
classification.SVM_kernel_test(X, Y)
Y_svm_pred = classification.SVM_classification(Xt, Yt, Xe)

# RF classification
print('Start RF classification')
classification.RF_parameter_test(X, Y)
Y_rf_pred = classification.RF_classification(Xt, Yt, Xe)

# Evaluate results
print('Start evaluating the result')
classification.Evaluation(Y_svm_pred, Ye)
classification.Evaluation(Y_rf_pred, Ye)
