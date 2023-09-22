
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import  datasets, svm
import matplotlib.pyplot as plt


# We will put all  utils here

# 1. Read digit
def read_digits():
    digits = datasets.load_digits()
    X= digits.images
    y=digits.target
    return X, y

# 2. Preprocess data

def preprocess_data(data):
# flatten the images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data


# 3. Split data into 50% train and 50% test subsets
def split_data(X,y,test_size,random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5,  random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# train the model
def train_model(X, y, model_params, model_type="svm"):
    if model_type == "svm":
         clf= svm.SVC;
    model = clf(**model_params)
    #Train the model
    
    model.fit(X,y)
    return model


def split_train_dev_test(X, y, test_size, dev_size):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    
    # Split the test set into development and test sets
    X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=dev_size, random_state=1)
    
    return X_train, X_dev, X_test, y_train, y_dev, y_test


def predict_evaluate(model,X_test,y_test):
    predicted=model.predict(X_test)
    print(
    f"Classification report for classifier {model}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
    )
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    plt.show()
    return predicted