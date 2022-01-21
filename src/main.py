from sympy import im
from train.train_model import TrainModel

if __name__ == "__main__":
    train_model = TrainModel()
    train_model.load()

    # model_list = ["BernoulliNB", "MultinomialNB", "SVC", "GaussianNB"]
    test_data = "今天天气真不错， 你觉得呢"
    model_list = ["GaussianNB"]
    for model_type in model_list:
        train_model.train(model_type)

    # train_model.predict(test_data)

