# Import datasets, classifiers and performance metrics
from sklearn import svm,datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from itertools import product

#read gigits
def read_digits():
    digits = datasets.load_digits()
    x = digits.images
    y = digits.target 
    return x,y

# We will define utils here :
def preprocess_data(data):
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

# Split data into 50% train and 50% test subsets

def split_data(X,y,test_size=0.5,random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=test_size, shuffle=False
    )
    return X_train, X_test, y_train, y_test

# Create a classifier: a support vector classifier
def train_model(X, y, model_params,model_type = 'svm'):
    if model_type == 'svm':
        clf = svm.SVC(**model_params)
    clf.fit(X, y)
    return clf

def split_train_dev_test(X, y, test_size, dev_size):
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
    X_train, X_dev, y_train, y_dev = split_data(X_train, y_train, test_size=dev_size)
    return X_train, X_test,X_dev, y_train, y_test, y_dev

def predict_and_eval(model, X_test, y_test):
    
    predicted = model.predict(X_test)

    # _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    # for ax, image, prediction in zip(axes, X_test, predicted):
    #     ax.set_axis_off()
    #     image = image.reshape(8, 8)
    #     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    #     ax.set_title(f"Prediction: {prediction}")
    accuracy = accuracy_score(y_test, predicted)

    # print(
    # f"Classification report for classifier {model}:\n"
    # f"{metrics.classification_report(y_test, predicted)}\n"
    # )

    # disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    # disp.figure_.suptitle("Confusion Matrix")
    # print(f"Confusion matrix:\n{disp.confusion_matrix}")
    # plt.show()
    return accuracy,predicted

def make_param_combinations(param_list_dict):
    hparams = param_list_dict.keys()
    ranges = [param_list_dict[x] for x in hparams]
    list_of_all_param_combination=[ dict(zip(hparams,x)) for x in list(product(*ranges))]
    return list_of_all_param_combination    
        

def tune_hparams(X_train, y_train, X_dev, y_dev, param_list_dict):
    list_of_all_param_combination = make_param_combinations(param_list_dict)
    best_accuracy_so_far = -1
    best_model = None
    for params in list_of_all_param_combination:
        cur_model = train_model(X_train, y_train, model_params=params, model_type='svm')
        cur_accuracy,predicted = predict_and_eval(cur_model, X_dev, y_dev)
        if cur_accuracy > best_accuracy_so_far:
            best_accuracy_so_far = cur_accuracy
            best_model = cur_model

    best_accuracy = best_accuracy_so_far
    best_hparams = best_model.get_params()
    return best_hparams, best_model, best_accuracy