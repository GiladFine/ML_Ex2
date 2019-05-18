import random
import sys
from collections import Counter

import numpy as np

CLASSES = ["0", "1", "2"]
FOLDS = 7
ITERATIONS = 20
LEARNING_RATE = 0.1
BIAS = 1

class FileReader(object):
    def __init__(self, data_file, classes_file):
        self.data = [line.split(",") for line in data_file.read().splitlines()]
        self.classes = [str(x[0]) for x in classes_file.read().splitlines()]
        #print(self.data)
        #print(self.classes)
        self.normalize()

    def normalize(self):
        nominal_data = [line[0] for line in self.data]
        nominal_data_counter = Counter(nominal_data)
        self.data = [[round(nominal_data_counter[line[0]] / len(self.data), 4)] + line[1:] for line in self.data]
        self.data = [[line[0]] + list(map(float, line[1:])) for line in self.data]
        #print(self.data)

class MultiClassPerceptron(object):

    def __init__(self, classes, data, folds, iterations):
        self.classes = classes
        self.data = [[data_item[0], data_item[1] + [BIAS]] for data_item in data]
        self.num_of_features = len(self.data[0][1]) - 1
        self.folds = folds
        self.iterations = iterations

        # Split feature data into train set, and test set
        k, m = divmod(len(self.data), FOLDS)
        self.folded_data = list(self.data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(FOLDS))

        self.train_set = [y for x in self.folded_data[1:] for y in x]
        self.test_set = self.folded_data[0]

        self.weights_arrays = {c: np.array([0 for _ in range(self.num_of_features + 1)]) for c in self.classes}


    def predict(self, feature_input_data):
        feature_input_array = np.array(feature_input_data)

        max_activation = 0
        prediction = self.classes[0]

        for c in self.classes:
            activation = np.dot(feature_input_array, self.weights_arrays[c])
            if activation >= max_activation:
                max_activation = activation
                prediction = c

        return prediction

    def train(self):
        for iter_num in range(self.iterations):
            for category, feature_data in self.train_set:
                feature_array = np.array(feature_data)

                max_activation = 0
                prediction = self.classes[0]

                # Multi-Class Decision Rule:
                for c in self.classes:
                    activation = np.dot(feature_array, self.weights_arrays[c])
                    if activation >= max_activation:
                        max_activation = activation
                        prediction = c

                # Update Rule:
                if not (category == prediction):
                    self.weights_arrays[category] = np.array([(x + LEARNING_RATE * y) for (x, y) in zip(self.weights_arrays[category], feature_array)])
                    self.weights_arrays[prediction] = np.array([(x - LEARNING_RATE * y) for (x, y) in zip(self.weights_arrays[prediction], feature_array)])

    def cross_validate(self):
        self.train()
        accuracy_avg = self.accuracy()
        for i in range(1, FOLDS):
            self.train_set += self.test_set
            self.test_set = self.folded_data[i]
            self.train_set = [x for x in self.train_set if x not in self.test_set]
            self.train()
            accuracy_avg += self.accuracy()
        accuracy_avg /= FOLDS
        print("AVG Accuracy:", accuracy_avg)

    def accuracy(self):
        correct = 0
        incorrect = 0
        for category, feature_data in self.test_set:
            prediction = self.predict(feature_data)
            if category == prediction:
                correct += 1
            else:
                incorrect += 1

        accuracy = (correct * 1.0) / ((correct + incorrect) * 1.0)
        print ("ACCURACY:")
        print ("Model Accuracy:", accuracy)
        return accuracy

class SVM:
    def __init__(self, train_x, train_y, epochs, eta, Lambda, k):
        self.train_x = train_x
        self.train_y = train_y
        self.epochs = epochs
        self.eta = eta
        self.Lambda = Lambda
        self.k = k

    def train(self):
        # Support Vector Machine algorithm
        N = len(self.train_x)
        n = len(self.train_x[0])
        w = np.zeros((self.k, n))
        for ep in range(self.epochs):
            arr = np.arange(N)
            np.random.shuffle(arr)
            self.train_x = self.train_x[arr]
            self.train_y = self.train_y[arr]
            for i in range(N):
                x = self.train_x[i]
                y = self.train_y[i]
                y = int(y)
                values = np.dot(w, x)
                # Put -inf in y index to find argmax without y
                values[y] = - np.inf
                y_hat = np.argmax(values)
                s = 1 - self.eta * self.Lambda
                for l in range(self.k):
                    if l == y:
                        w[l, :] = s * w[l, :] + self.eta * x
                    elif l == y_hat:
                        w[l, :] = s * w[l, :] - self.eta * x
                    else:
                        w[l, :] = s * w[l, :]
        return w

    def evaluate(self, dev_x, dev_y, w):
        # compute dev accuracy using trained parameters
        accuracy = 0
        N = len(dev_x)
        for x, y in zip(dev_x, dev_y):
            values = np.dot(w, x)
            y_hat = np.argmax(values)
            if y_hat == y:
                accuracy += 1
        accuracy /= N
        return accuracy



def load_data(data_file):
    data = []
    # Sex feature to numerical index dictionary
    sex_to_index = {'M': 0, 'F': 1, 'I': 2}
    # Read file
    with open(data_file, 'r') as file:
        for line in file:
            # Split entry
            line = line.strip().split(',')
            line[0] = sex_to_index[line[0]]
            # Add to data
            data.append(np.array(line, dtype=np.float64))
    return np.array(data)

def load_labels(labels_file):
    # Read labels from file
    labels = np.loadtxt(labels_file, dtype=np.float64)
    return labels

def main():
    data_path = 'train_x.txt' #sys.argv[1]
    classes_path = 'train_y.txt' #sys.argv[2]

    with open(data_path, "r") as data_file, open(classes_path, "r") as classes_file:
        reader = FileReader(data_file, classes_file)

    data = list(zip(reader.classes, reader.data))
    random.shuffle(data)

    perceptron = MultiClassPerceptron(CLASSES, data, FOLDS, ITERATIONS)
    perceptron.cross_validate()

    svm = SVM(load_data(data_path), load_labels(classes_path), epochs=100, eta=0.01, Lambda=0.5, k=3)
    svm.train()


if __name__ == "__main__":
    main()
