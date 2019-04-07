
# parser for any file
# read file and save output in 2D array
# @return: 2D array
def readFile(filePath):
    with open(filePath, 'r') as file:
        line_array = file.read().splitlines()
        cell_array = [line.split() for line in line_array]
        file.close()
        return cell_array


# store data in file
def writeFile(filePath, data, append=False):
    if not append:
        with open(filePath, 'w+') as file:
            for element in data:
                file.write(str(element) + '\t')
    else:
        with open(filePath, 'a+') as file:
            file.write('\n')
            for element in data:
                file.write(str(element) + '\t')
    file.close()
