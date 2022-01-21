import os
import jieba


def get_files_path(file_dir, filetype=".txt"):
    """得到文件夹下的所有.txt文件的路径
    Args:
        file_dir: 文件夹路径
        filetype: 文件后缀
    Returns:
        所有filetype类型文件的绝对路径
    """
    files_path = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if filetype is None or (os.path.splitext(file)[1] == filetype):
                files_path.append(os.path.join(root, file))
    return files_path


def load_stopwords():
    path = "../data/stopwords/cn_stopwords.txt"
    with open(path, "r") as f:
        stopwords = f.read().split("\n")
    return stopwords


def get_train_data_dir():
    return "../data/dataset/train"


def get_test_data_dir():
    return "../data/dataset/test"


def get_tfidf_model_path():
    return "../data/model/tfidf"


def get_cls_model_path():
    return "../data/model/cls_model"


model_dir = "../data/model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
