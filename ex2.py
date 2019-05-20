import random
import sys
from collections import Counter

import numpy as np

CLASSES = [0, 1, 2]
BIAS = 1

class FileReader(object):
    def __init__(self, data_file, classes_file, test_file):
        self.train = [line.split(",") for line in data_file.read().splitlines()]
        self.classes = [int(x[0]) for x in classes_file.read().splitlines()]
        self.test = [line.split(",") for line in test_file.read().splitlines()]

        self.train = self.normalize(self.train)
        self.test = self.normalize(self.test)

    def normalize(self, data):
        # Nominal data
        nominal_data = [line[0] for line in data]
        nominal_data_counter = Counter(nominal_data)
        data = [[round(nominal_data_counter[line[0]] / len(data), 4)] + line[1:] for line in data]
        data = [[line[0]] + list(map(float, line[1:])) for line in data]

        # Other data
        max_values = [0 for _ in range(len(data[0]) - 1)]
        min_values = [0 for _ in range(len(data[0]) - 1)]
        for i in range(len(max_values)):
            max_values[i] = (max(np.array(data)[:,i + 1]))
            min_values[i] = (min(np.array(data)[:,i + 1]))

        for i in range(len(data)):
            for j in range(len(data[i]) - 1):
                if max_values[j] - min_values[j] == 0:
                    max_values[j] += 0.01

            data[i] = [data[i][0]] + [(data[i][j + 1] - min_values[j]) / (max_values[j] - min_values[j]) for j in range(len(data[i]) - 1)]
        return data


class MLAlgorithm(object):

    def cross_validate(self):
        self.train()
        accuracy_avg = self.accuracy()
        for i in range(1, self.folds):
            self.train_set += self.test_set
            self.test_set = self.folded_data[i]
            self.train_set = [x for x in self.train_set if x not in self.test_set]
            self.train()
            accuracy_avg += self.accuracy()
        accuracy_avg /= self.folds
        print("AVG Accuracy:", accuracy_avg)

    def predict(self, test_x):
        N = len(test_x)
        y_hats = np.zeros(N, dtype=int)
        for i in range(N):
            x = test_x[i]
            values = np.dot(self.w, x)
            y_hat = np.argmax(values)
            y_hats[i] = y_hat
        return y_hats

    def accuracy(self):
        correct = 0
        incorrect = 0
        results = self.predict(self.test_x)
        for i in range(len(results)):
            if results[i] == self.test_y[i]:
                correct += 1
            else:
                incorrect += 1

        accuracy = (correct * 1.0) / ((correct + incorrect) * 1.0)
        print("Model Accuracy:", accuracy)
        return accuracy

class MultiClassPerceptron(MLAlgorithm):

    def __init__(self, classes, data, iterations, folds, eta):
        self.classes = classes
        self.data = [[data_item[0], data_item[1] + [BIAS]] for data_item in data]
        self.num_of_features = len(self.data[0][1]) - 1
        self.folds = folds
        self.eta = eta
        self.iterations = iterations

        # Split feature data into train set, and test set
        k, m = divmod(len(self.data), self.folds)
        self.folded_data = list(self.data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(self.folds))

        self.train_set = self.data
        #self.train_set = [y for x in self.folded_data[1:] for y in x]
        #self.test_set = self.folded_data[0]

        self.train_x = [line[1] for line in self.train_set]
        self.train_y = [line[0] for line in self.train_set]

        #self.test_x = [line[1] for line in self.test_set]
        #self.test_y = [line[0] for line in self.test_set]

        self.w = np.zeros((len(self.classes), len(self.train_x[0])))

    def train(self):
        self.w = np.zeros((len(self.classes), len(self.train_x[0])))
        for iter_num in range(self.iterations):
            random.seed(315)
            random.shuffle(self.train_set)
            for category, feature_data in self.train_set:
                feature_array = np.array(feature_data)

                max_activation = 0
                prediction = self.classes[0]

                # Multi-Class Decision Rule:
                for i in range(len(self.classes)):
                    activation = np.dot(feature_array, self.w[i])
                    if activation >= max_activation:
                        max_activation = activation
                        prediction = self.classes[i]

                # Update Rule:
                if not (category == prediction):
                    class_index = self.classes.index(category)
                    pred_index = self.classes.index(prediction)
                    self.w[class_index] = np.array([(x + self.eta * y) for (x, y) in zip(self.w[class_index], feature_array)])
                    self.w[pred_index] = np.array([(x - self.eta * y) for (x, y) in zip(self.w[pred_index], feature_array)])

class SVM(MLAlgorithm):
    def __init__(self, classes, data, iterations, folds, eta, Lambda):

        # Split feature data into train set, and test set
        self.classes = classes
        self.data = data
        self.iterations = iterations
        self.folds = folds
        self.eta = eta
        self.Lambda = Lambda

        k, m = divmod(len(self.data), self.folds)
        self.folded_data = list(self.data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(self.folds))

        self.train_set = self.data
        #self.train_set = [y for x in self.folded_data[1:] for y in x]
        #self.test_set = self.folded_data[0]

        self.train_x = [line[1] for line in self.train_set]
        self.train_y = [line[0] for line in self.train_set]

        #self.test_x = [line[1] for line in self.test_set]
        #self.test_y = [line[0] for line in self.test_set]

        self.w = np.zeros((len(self.classes), len(self.train_x[0])))


    def train(self):
        # Support Vector Machine algorithm
        N = len(self.train_x)
        n = len(self.train_x[0])
        self.w = np.zeros((len(self.classes), n))
        for iter in range(self.iterations):
            random.seed(3044)
            random.shuffle(self.train_set)
            self.train_x = [line[1] for line in self.train_set]
            self.train_y = [line[0] for line in self.train_set]
            for i in range(N):
                x = self.train_x[i]
                y = self.train_y[i]
                y = int(y)
                values = np.dot(self.w, x)
                # Put -inf in y index to find argmax without y
                values[y] = - np.inf
                y_hat = np.argmax(values)
                s = 1 - self.eta * self.Lambda
                for c in self.classes:
                    if c == y:
                        self.w[c, :] = [(s * wi + self.eta * xj) for wi, xj in zip(self.w[c, :], x)]
                    elif c == y_hat:
                        self.w[c, :] = [(s * wi - self.eta * xj) for wi, xj in zip(self.w[c, :], x)]
                    else:
                        self.w[c, :] = [s * i for i in self.w[c, :]]

class PA(MLAlgorithm):
    def __init__(self, classes, data, iterations, folds):
        # Split feature data into train set, and test set
        self.classes = classes
        self.data = data
        self.iterations = iterations
        self.folds = folds

        k, m = divmod(len(self.data), self.folds)
        self.folded_data = list(self.data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(self.folds))

        self.train_set = self.data
        #self.train_set = [y for x in self.folded_data[1:] for y in x]
        #self.test_set = self.folded_data[0]

        self.train_x = [line[1] for line in self.train_set]
        self.train_y = [line[0] for line in self.train_set]

        #self.test_x = [line[1] for line in self.test_set]
        #self.test_y = [line[0] for line in self.test_set]

        self.w = np.zeros((len(self.classes), len(self.train_x[0])))

    def train(self):
        #PA algorithm
        N = len(self.train_x)
        n = len(self.train_x[0])
        self.w = np.zeros((len(self.classes), n))
        for iter in range(self.iterations):
            random.seed(3044)
            random.shuffle(self.train_set)
            self.train_x = [line[1] for line in self.train_set]
            self.train_y = [line[0] for line in self.train_set]
            for i in range(N):
                x = self.train_x[i]
                y = self.train_y[i]
                y = int(y)
                values = np.dot(self.w, x)
                # Put -inf in y index to find argmax without y
                values[y] = - np.inf
                y_hat = np.argmax(values)
                # compute tau
                err = 1 - np.dot(self.w[y, :], x) + np.dot(self.w[y_hat, :], x)
                loss = np.max([0, err])
                tau = loss / np.dot(x, x)
                # update w
                for c in self.classes:

                    if c == y:
                        self.w[c, :] = [(wi + tau * xj) for wi, xj in zip(self.w[c, :], x)]

                    if c == y_hat:
                        self.w[c, :] = [(wi - tau * xj) for wi, xj in zip(self.w[c, :], x)]
def main():
    data_path = 'train_x_bla.txt' #sys.argv[1]
    classes_path = 'train_y_bla.txt' #sys.argv[2]
    test_path = 'test_x_bla.txt' #sys.argv[2]

    with open(data_path, "r") as data_file, open(classes_path, "r") as classes_file, open(test_path, "r") as test_file:
        reader = FileReader(data_file, classes_file, test_file)

    data = list(zip(reader.classes, reader.train))

    perceptron = MultiClassPerceptron(CLASSES, data, iterations=25, folds=4, eta=0.2)
    perceptron.train()

    svm = SVM(CLASSES, data, iterations=40, folds=4, eta=0.01, Lambda=0.5)
    svm.train()

    pa = PA(CLASSES, data, iterations=100, folds=4)
    pa.train()

    y_hats_svm = svm.predict(reader.test)
    y_hats_pa = pa.predict(reader.test)

    reader.test = [item + [BIAS] for item in reader.test]
    y_hats_per = perceptron.predict(reader.test)

    # print predictions
    for i in range(len(reader.test)):
        print('perceptron: ' + str(y_hats_per[i]) + ', ' + 'svm: ' +
              str(y_hats_svm[i]) + ', ' 'pa: ' + str(y_hats_pa[i]))

if __name__ == "__main__":
    main()