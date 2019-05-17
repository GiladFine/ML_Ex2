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
        print(self.data)
        print(self.classes)
        self.normalize()

    def normalize(self):
        nominal_data = [line[0] for line in self.data]
        nominal_data_counter = Counter(nominal_data)
        self.data = [[round(nominal_data_counter[line[0]] / len(self.data), 4)] + line[1:] for line in self.data]
        self.data = [[line[0]] + list(map(float, line[1:])) for line in self.data]
        print(self.data)

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


def main():
    data_path = sys.argv[1]
    classes_path = sys.argv[2]
    with open(data_path, "r") as data_file, open(classes_path, "r") as classes_file:
        reader = FileReader(data_file, classes_file)

    data = list(zip(reader.classes, reader.data))
    random.shuffle(data)
    perceptron = MultiClassPerceptron(CLASSES, data, FOLDS, ITERATIONS)
    perceptron.cross_validate()

if __name__ == "__main__":
    main()
