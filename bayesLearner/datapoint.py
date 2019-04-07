import numpy as np

class Datapoint:

    def __init__(self, featureVector, classValue):
        self.featureVector = np.array(featureVector)
        self.classValue = classValue

    def numOfFeatures(self):
        return len(self.featureVector)

    def print(self):
        print('features: ', end=" ")
        if type(self.featureVector) == str:
            print(self.featureVector)
        else:
            print(*self.featureVector, sep=", ", end=" ")
        print('\tclass: ', end=" ")
        print(self.classValue)
