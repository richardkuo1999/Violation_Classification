import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,\
                recall_score, f1_score, multilabel_confusion_matrix, top_k_accuracy_score
np.seterr(divide='ignore',invalid='ignore')

"""
ConfusionMetric
Mertric   P    N
P        TP    FN
N        FP    TN
"""
# TODO TOP1
class ClassifyMetric(object):
    def __init__(self, numClass, labels=None):
        self.labels = labels
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)
        
    def genConfusionMatrix(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred, labels=self.labels)

    def addBatch(self, y_true, y_pred):
        assert  np.array(y_true).shape == np.array(y_pred).shape
        self.confusionMatrix += self.genConfusionMatrix(y_true, y_pred)
 
    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

    def accuracy(self):
        accuracy = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return accuracy

    def precision(self):
            precision = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
            return np.nan_to_num(precision)

    def recall(self):
        recall = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return recall

    def f1_score(self):
        precision = self.precision()
        recall = self.recall()
        f1_score = 2 * (precision*recall) / (precision+recall)
        return np.nan_to_num(f1_score)

if __name__ == '__main__':
    y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    metric = ClassifyMetric(3, ["ant", "bird", "cat"])
    metric.addBatch(y_true, y_pred)
    acc = metric.accuracy()
    precision = metric.precision()
    recall = metric.recall()
    f1Score = metric.f1_score()
    print('acc is : %f' % acc)
    print('precision is :', precision)
    print('recall is :', recall)
    print('f1_score is :', f1Score)


