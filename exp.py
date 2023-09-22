from utils import preprocess_data,split_data,train_model,read_digits,predict_evaluate
# 1. Load the data
X, y = read_digits()

# 2. Split the data
X_train, X_test, y_train, y_test=split_data(X, y,test_size=0.3)

# 3. Data Preprocessing

X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)

# 4. train the data
model=train_model(X_train, y_train, {'gamma': 0.001}, model_type="svm")

# 5.Prediction and evaluation
predicted = predict_evaluate(model, X_test, y_test)