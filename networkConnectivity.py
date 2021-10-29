import numpy as np


def createPatterns(n, number_of_patterns):
    m = np.random.choice([-1, 1], (number_of_patterns, n))
    w = np.zeros((n, n))
    return m, w


def modifyRandomlyAPattern(m, numberOfValuesToChange):
    number_of_patterns, n = m.shape
    patternToModify = np.random.randint(0, number_of_patterns)
    changedIndexes = []
    i = 0
    while i != numberOfValuesToChange:
        randomPosition = np.random.randint(0, n)
        if not changedIndexes.count(randomPosition) > 0:
            m[patternToModify, randomPosition] = -1 * m[patternToModify, randomPosition]
            changedIndexes.append(randomPosition)
            i += 1
    return m, patternToModify


def retrievingMemorizedPattern(unmodifiedPatternList, modifiedPatternList, modifiedPattern, w, numberOfIterations):
    for i in range(numberOfIterations):
        newPattern = np.matmul(w, modifiedPatternList[modifiedPattern])
        for j in range(len(newPattern)):
            if newPattern[j] < 0:
                newPattern[j] = -1
            else:
                newPattern[j] = 1
        modifiedPatternList[modifiedPattern] = newPattern
        if (unmodifiedPatternList[modifiedPattern] == modifiedPatternList[modifiedPattern]).all():
            break
    return modifiedPatternList


def memorizePatterns(m, w):
    number_of_patterns, n = m.shape
    rows, columns = w.shape
    for i in range(rows):
        for j in range(columns):
            matrix_element = 0
            for pattern in range(number_of_patterns):
                if i < j:
                    matrix_element += m[(pattern, i)] * m[(pattern, j)]
                w[(i, j)] = matrix_element / number_of_patterns
    w += w.T

    return w
