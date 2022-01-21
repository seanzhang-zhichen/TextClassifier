import os
import re
import joblib
import jieba
import logging
import random
import emoji
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, CategoricalNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier

from sklearn import metrics
from sklearn.model_selection import train_test_split

from utils.common import get_train_data_dir
from utils.common import get_test_data_dir

from utils.common import get_files_path
from utils.common import load_stopwords

from utils.common import get_tfidf_model_path
from utils.common import get_cls_model_path


class TrainModel:
    def __init__(self):
        self.train_data_dir = get_train_data_dir()
        self.test_data_dir = get_test_data_dir()
        self.stopwords = load_stopwords()
        self.tfidf_path = get_tfidf_model_path()
        self.model_path = get_cls_model_path()

    def load(self):
        if not os.path.exists(self.model_path):
            print("模型不存在，开始训练模型...")
            self.train()
        print("加载模型...")
        self.tf_idf = joblib.load(self.tfidf_path)
        self.model = joblib.load(self.model_path)

    def load_data(self, data_dir):
        """加载文件内容和标签"""
        files = get_files_path(data_dir, ".txt")
        contents = []
        labels = []
        for file in files:
            with open(file, "r") as f:
                data = f.read()
            data_cut = " ".join(jieba.cut(data))
            contents.append(data_cut)
            label = file.split("/")[-2]
            labels.append(label)
        return contents, labels

    def train(self, model_type="BernoulliNB"):
        print("开始训练 {} 模型".format(model_type))
        X_train, y_train = self.load_data(self.train_data_dir)
        test_content, test_labels = self.load_data(self.test_data_dir)
        tfidf = TfidfVectorizer(stop_words=self.stopwords)
        train_data = tfidf.fit(X_train)
        train_data = tfidf.transform(X_train)
        test_data = tfidf.transform(test_content)
        # print("train_data: {}".format(train_data.shape))
        # print("test_data: {}".format(test_data.shape))

        joblib.dump(tfidf, self.tfidf_path, compress=1)

        print("保存tfidf成功，目标路径: {}".format(self.tfidf_path))
        if model_type == "BernoulliNB":
            model = BernoulliNB()
        elif model_type == "MultinomialNB":
            model = MultinomialNB()
        elif model_type == "CategoricalNB":
            model = CategoricalNB()
            train_data = train_data.toarray()
            test_data = test_data.toarray()
        elif model_type == "GaussianNB":
            model = GaussianNB()
        elif model_type == "SVC":
            model = SVC(probability=True)

        # print("train_data: {}".format(train_data.shape))
        # print("test_data: {}".format(test_data.shape))

        model.fit(train_data, y_train)

        joblib.dump(model, self.model_path, compress=1)
        print("保存 {} 模型成功，目标路径: {}".format(model_type, self.model_path))

        predict_test = model.predict(test_data)

        print("----" * 20)
        print(
            "{} 模型准确率为: {}".format(
                model_type, metrics.accuracy_score(test_labels, predict_test)
            )
        )

        print("----" * 20)

    def predict_prob(self, test_data):
        result = {}
        test_data = " ".join(jieba.cut(test_data))
        test_vec = self.tf_idf.transform([test_data])
        scores = self.model.predict_proba(test_vec)[0]
        for idx, label in enumerate(self.model.classes_):
            result[label] = scores[idx]
        return result

    def predict(self, test_data):
        test_data = " ".join(jieba.cut(test_data))
        test_vec = self.tf_idf.transform([test_data])
        res = self.model.predict(test_vec)[0]
        return res

