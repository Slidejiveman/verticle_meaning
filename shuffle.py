import random

### This script reads in the dataset, which was created with the help of ChatGPT 4.
### It then shuffles the dataset so that the lines are not left in discernible groups.

## Test Set
with open("./datasets/original/testset.txt", 'r') as file:
    lines = file.readlines()

if lines is not None:
    random.shuffle(lines)
else:
    lines = ["The input file was empty."]

with open("./datasets/shuffled/testsetshuff.txt", 'w') as output:
    output.writelines(lines)
    
## Validation Set
with open("./datasets/original/valset.txt", 'r') as file:
    lines = file.readlines()

if lines is not None:
    random.shuffle(lines)
else:
    lines = ["The input file was empty."]

with open("./datasets/shuffled/valsetshuff.txt", 'w') as output:
    output.writelines(lines)
    
## Training Set
with open("./datasets/original/trainset.txt", 'r') as file:
    lines = file.readlines()

if lines is not None:
    random.shuffle(lines)
else:
    lines = ["The input file was empty."]

with open("./datasets/shuffled/trainsetshuff.txt", 'w') as output:
    output.writelines(lines)