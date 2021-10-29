import networkConnectivity as nc

# After 3.2 what the program should do

# Create 3 patterns of size 50
patternList, w = nc.createPatterns(50, 3)
# Update the w matrix and memorises the 3 patterns with Heb05
w = nc.memorizePatterns(patternList, w)
print(w)


#  After 3.3 what the program should do

# Create 3 patterns of size 50
patternList, w = nc.createPatterns(50, 3)
# Update the w matrix and memorises the 3 patterns with Heb05
w = nc.memorizePatterns(patternList, w)
# Modify randomly a pattern from the given patternList and changes 3 values.
# Return the modified pattern list in newPatternList and it's index in modifiedPattern
newPatternList, modifiedPattern = nc.modifyRandomlyAPattern(patternList, 3)
# Retrieve pattern list if possible or after 20 iterations
guessedPatterns = nc.retrievingMemorizedPattern(patternList, newPatternList, modifiedPattern, w, 20)
# Test if the patterns are the same
print((guessedPatterns == patternList).all())

#