import datapoint as dp
import numpy as np

class Dataset:

    def __init__(self, datapoints=[]):
        self.dataset = datapoints

    def addDatapoint(self, datapoint):
        self.dataset.append(datapoint)

    def addRawMatrix(self, matrix):
        for row in matrix:
            self.dataset.append(dp.Datapoint(
                row[1:len(row)],
                row[0]))

    # maps all datapoint values to integers
    def mapDatapointsToNumValues(self):
        for i in range(0, len(self.dataset)):
            if (self.dataset[i].classValue == 'A'):
                self.dataset[i].classValue = 0
            else:
                self.dataset[i].classValue = 1
            self.dataset[i].featureVector = self.dataset[i].featureVector.astype(np.float64)

    def print(self):
        for datapoint in self.dataset:
            datapoint.print()

    def numOfFeatures(self):
        return self.dataset[0].numOfFeatures()

    def length(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]

    def set(self, idx, datapoint):
        if idx >= 0 & len(self.dataset) > idx:
            self.dataset[idx] = datapoint

    def isEmpty(self):
        return len(self.dataset) == 0

    def clear(self):
        self.dataset = []

    def extractSubset(self, lambdaFunc):
        subset = Dataset([])
        for datapoint in self.dataset:
            if lambdaFunc(datapoint):
                subset.addDatapoint(datapoint)
        return subset

    # @return: amount of instances belonging to certain class
    def amountOfInstancesForClass(self, classValue):
        amount = 0
        for datapoint in self.dataset:
            if datapoint.classValue == classValue:
                amount += 1
        return amount

    def get_column(self, column_number):
        column_set = []
        for datapoint in self.dataset:
            if column_number == 0:
                column_set.append(datapoint.classValue)
            else:
                column_set.append(datapoint.featureVector[column_number - 1])
        return column_set

    def extract_class_points(self, class_value):
        class_set = []
        for datapoint in self.dataset:
            if datapoint.classValue == class_value:
                class_set.append(datapoint.featureVector)
        return class_set
