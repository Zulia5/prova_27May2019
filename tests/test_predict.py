from newdir import predict
from keras.datasets import boston_housing

def test_predict():
    _, (x, _) = boston_housing.load_data()
    predict(x)
