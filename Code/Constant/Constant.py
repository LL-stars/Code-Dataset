Dpath = './Dataset'

datasets = {
    'assist0910' : 'assist0910',
    'assist2017' : 'assist2017',
    'Eedi' : 'Eedi',
    'SCD' : 'SCD'
}

# question number of each dataset
numbers = {
    'assist0910' : 124,  
    'assist2017' : 102,
    'Eedi' : 948,
    'SCD' : 45
}
DATASET = datasets['assist0910']
NUM_OF_QUESTIONS = numbers['assist0910']
# the max step of RNN model
MAX_STEP = 100
BATCH_SIZE = 64
LR = 0.001
EPOCH = 100
#input dimension
INPUT = NUM_OF_QUESTIONS * 2
# embedding dimension
EMBED = NUM_OF_QUESTIONS
# hidden layer dimension
HIDDEN = 200
# nums of hidden layers
LAYERS = 1
# output dimension
OUTPUT = NUM_OF_QUESTIONS
