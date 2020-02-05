import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluator(object):
    def __init__(self, predictions, labels, average="macro"):
        self.predictions = predictions
        self.labels = labels
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.average = average

    def is_better_than(self, evaluator, metrics):
        if "accuracy" in metrics and self.accuracy is not None and evaluator.accuracy is not None \
                and evaluator.accuracy > self.accuracy:
            return False
        if "precision" in metrics and self.precision is not None and evaluator.precision is not None \
                and evaluator.precision > self.precision:
            return False
        if "recall" in metrics and self.recall is not None and evaluator.recall is not None \
                and evaluator.recall > self.recall:
            return False
        if "f1" in metrics and self.f1 is not None and evaluator.f1 is not None \
                and evaluator.f1 > self.f1:
            return False
        return True

    def evaluate(self, metrics):
        eval_result = dict()
        if "accuracy" in metrics:
            print("Accuracy score: {}".format(self._compute_accuracy()))
            eval_result["accuracy"] = np.mean(self._compute_accuracy())
        if "precision" in metrics:
            print("Precision score: {}".format(self._compute_precision()))
            eval_result["precision"] = np.mean(self._compute_precision())
        if "recall" in metrics:
            print("Recall score: {}".format(self._compute_recall()))
            eval_result["recall"] = np.mean(self._compute_recall())
        if "f1" in metrics:
            print("F1 score: {}".format(self._compute_f1()))
            eval_result["f1"] = np.mean(self._compute_f1())
        return eval_result

    def _compute_accuracy(self):
        self.accuracy = accuracy_score(y_true=self.labels, y_pred=self.predictions) \
            if self.accuracy is None else self.accuracy
        return self.accuracy

    def _compute_f1(self):
        self.f1 = f1_score(y_true=self.labels, y_pred=self.predictions, average=self.average) \
            if self.f1 is None else self.f1
        return self.f1

    def _compute_precision(self):
        self.precision = precision_score(y_true=self.labels, y_pred=self.predictions, average=self.average) \
            if self.precision is None else self.precision
        return self.precision

    def _compute_recall(self):
        self.recall = recall_score(y_true=self.labels, y_pred=self.predictions, average=self.average) \
            if self.recall is None else self.recall
        return self.recall
