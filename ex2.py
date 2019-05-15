import random
import sys
from collections import Counter

import numpy as np

TEST_TRAIN_RATIO = 0.85
ITERATIONS = 100
BIAS = 1

class FileReader(object):
    def __init__(self, input_file):
        self.file = input_file
        self.data = [line.split(",") for line in self.file.read().splitlines()]
        print(self.data)
        self.normalize()

    def normalize(self):
        nominal_data = [line[0] for line in self.data]
        nominal_data_counter = Counter(nominal_data)
        self.data = [[round(nominal_data_counter[line[0]] / len(self.data), 4)] + line[1:] for line in self.data]
        self.data = [[line[0]] + list(map(float, line[1:])) for line in self.data]
        print(self.data)

class MultiClassPerceptron(object):

    def __init__(self, classes, features, data, train_test_ratio, iterations):
        self.classes = classes
        self.features = features
        self.data = data
        self.train_test_ratio = train_test_ratio
        self.iterations = iterations

        # Split feature data into train set, and test set
        random.shuffle(self.data)
        self.train_set = self.data[:int(len(self.data) * self.ratio)]
        self.test_set = self.data[int(len(self.data) * self.ratio):]

        self.weights_arrays = {c: np.array([0 for _ in range(len(features) + 1)]) for c in self.classes}

    def predict(self, feature_input_data):
        feature_input_data.append(BIAS)
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
        for _ in range(self.iterations):
            for category, feature_data in self.train_set:
                feature_data.append(BIAS)
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
                    self.weights_arrays[category] += feature_array
                    self.weights_arrays[prediction] -= feature_array

    def accuracy(self):
        correct = 0
        for category, feature_data in self.test_set:
            prediction = self.predict(feature_data)
            if category == prediction:
                correct += 1

        print ("ACCURACY:")
        print ("Model Accuracy:", (correct * 1.0) / ((len(feature_data)) * 1.0))


def main():
    input_path = sys.argv[1]
    with open(input_path, "r") as input_file:
        reader = FileReader(input_file)

if __name__ == "__main__":
    main()
