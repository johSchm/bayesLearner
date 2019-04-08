import dataset as ds
import fileParser as fp
import scatterplot as plot
import math


dataset = 0


# computes probability of class
# @return: prob
def computeClassProb(classValue):
    return dataset.amountOfInstancesForClass(classValue) / dataset.length()


# estimation
# computes mean / expectation of distribution
# of attribute values of instances belonging to a certain class
# @return: mean
def computeMean(classValue, attrIdx):
    attrValues = 0
    for i in range(0, dataset.length()):
        if dataset.get(i).classValue == classValue:
            attrValues += dataset.get(i).featureVector[attrIdx]
    return (1/dataset.amountOfInstancesForClass(classValue)) * attrValues


# estimation
# computes variance
# @return: variance
def computeVariance(classValue, attrIdx, mean):
    sse = 0
    for i in range(0, dataset.length()):
        if dataset.get(i).classValue == classValue:
            sse += (dataset.get(i).featureVector[attrIdx] - mean)**2
    return (1/(dataset.amountOfInstancesForClass(classValue)-1)) * sse


# probability of instance with a certain attribute value given a specific class
# @return: prob
def probOfAttrGivenClass(classValue, attrIdx, attrValue):
    mean = computeMean(classValue, attrIdx)
    variance = computeVariance(classValue, attrIdx, mean)
    return (1/math.sqrt(2*math.pi*variance)) * math.exp(-((attrValue - mean)**2)/(2*variance))


# probability of instance given a specific class
# @return: prob
def probOfInstanceGivenClass(datapoint, classValue):
    prop = 1
    for a in range(0, datapoint.featureVector.size):
        prop *= probOfAttrGivenClass(classValue, a, datapoint.featureVector[a])
    return prop


# computes the probability of a instance belonging to a certain class
# @return: prob
def probOfClassGivenInstance(datapoint, classValue):
    probXC = probOfInstanceGivenClass(datapoint, classValue)
    probC = computeClassProb(classValue)
    probTotal = 0
    for c in range(0, 2):
        probTotal += (probOfInstanceGivenClass(datapoint, c) * computeClassProb(c))
    return (probXC * probC) / probTotal


# naive bayes algorithm with gaussian estimation
# predict the class
# @return: class
def predictClass(datapoint):
    if probOfClassGivenInstance(datapoint, 0) > probOfClassGivenInstance(datapoint, 1):
        return 0
    else:
        return 1


# @return: boolean if the datapoint is classified correct
def predictionCorrect(datapoint):
    return datapoint.classValue == predictClass(datapoint)


# @return: boolean if the datapoint is classified correct
def predictionIncorrect(datapoint):
    return not datapoint.classValue == predictClass(datapoint)


# @return: dataset of misclassified datapoints
def extractMisses():
    return dataset.extractSubset(predictionIncorrect)


# @main
if __name__ == "__main__":

    # user input
    datafile = input("Input-file: ")

    print('Computing...')

    # convert to dataset
    rawDatamatrix = fp.readFile(datafile)
    dataset = ds.Dataset()
    dataset.addRawMatrix(rawDatamatrix)
    dataset.mapDatapointsToNumValues()

    # class 0 -> A
    for c in range(0, 2):
        print('\nResults for Class ' + c + ':')
        mean_attr0 = computeMean(c, 0)
        mean_attr1 = computeMean(c, 1)
        print('Mean of attribute 1: ' + computeMean(c, 0))
        print('Mean of attribute 2: ' + computeMean(c, 1))
        print('Variance of attribute 1: ' + computeVariance(c, 0, mean_attr0))
        print('Variance of attribute 1: ' + computeVariance(c, 1, mean_attr1))
        print('Class probability: ' + computeClassProb(c))

    plot.scatterplot(dataset[:, 1], dataset[:, 2], "x", "y", "data", "r")

    print('Done!')
