from jp_grammar_model import * # grammar types
from jp_grammar_utilities import diagramizer, quantizer, visualizer
from lambeq import Dataset, IQPAnsatz, PennyLaneModel, PytorchTrainer, UnifyCodomainRewriter
import logging
import numpy as np
import pennylane as qml
import random
import torch

## Setup logging
logging.basicConfig(filename='./logs/hybrid/app/app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

## Define loss and accuracy functions
def acc(y_hat, y):
    return (torch.argmax(y_hat, dim=1) ==
            torch.argmax(y, dim=1)).sum().item()/len(y)

def loss(y_hat, y):
    return torch.nn.functional.mse_loss(y_hat, y)

## Open datasets and collect sentences
def read_data(filename):
    labels, sentences = [], []
    with open(filename) as file:
        for line in file:
            t = int(line[0])
            labels.append([t, 1-t]) # returns an array with the correct label in first place
            sentences.append(line[1:].strip())
    return labels, sentences

## Read API key for IBM hardware
def read_key(filename):
    with open(filename) as file:
        key = file.readline().strip()
    return key

## define utilities and hyperparameters
BATCH_SIZE = 10 #2 gets 69% and 10 gets 60% on test sets
EPOCHS = 15
LEARNING_RATE = 0.1
SEED = 42
optimizer = torch.optim.AdamW
evaluate_functions = {'acc': acc}
with_real_hardware = False
log_dir = "./logs/hybrid"
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
u_cdom_rewriter = UnifyCodomainRewriter(s)
ansatz = IQPAnsatz({n: 1, s: 1, h: 1}, n_layers=2, n_single_qubit_params=3)

## Procedure
# read in data and labels
train_labels, train_data = read_data('./datasets/shuffled/trainsetshuff.txt')
val_labels, val_data = read_data('./datasets/shuffled/valsetshuff.txt')
test_labels, test_data = read_data('./datasets/shuffled/testsetshuff.txt')

# use diagramizer to build diagrams
train_diagrams = [diagramizer(sentence) for sentence in train_data if sentence is not None]
val_diagrams = [diagramizer(sentence) for sentence in val_data if sentence is not None]
test_diagrams = [diagramizer(sentence) for sentence in test_data if sentence is not None]
logging.info("Diagrams constructed from corpus data.")

# Rewrite diagrams so that they are the same shape and can be used with pytorch tensors.
train_diagrams = [u_cdom_rewriter(diagram) for diagram in train_diagrams if diagram is not None]
val_diagrams = [u_cdom_rewriter(diagram) for diagram in val_diagrams if diagram is not None]
test_diagrams = [u_cdom_rewriter(diagram) for diagram in test_diagrams if diagram is not None]
logging.info("Diagrams padded.")

# create labeled maps of diagrams (this is not using the read in data directly)
train_labels = [label for (diagram, label) in zip(train_diagrams, train_labels) if diagram is not None]
val_labels = [label for (diagram, label) in zip(val_diagrams, val_labels) if diagram is not None]
test_labels = [label for (diagram, label) in zip(test_diagrams, test_labels) if diagram is not None]
logging.info("Label lists constructed.")

# construct circuits
train_circuits = [quantizer(diagram, ansatz) for diagram in train_diagrams if diagram is not None]
val_circuits = [quantizer(diagram, ansatz) for diagram in val_diagrams if diagram is not None]
test_circuits = [quantizer(diagram, ansatz) for diagram in test_diagrams if diagram is not None]
logging.info("Circuits constructed from diagrams.")

# instantiate training model
all_circuits = train_circuits + val_circuits + test_circuits
if with_real_hardware:
    key = read_key('./API_key.txt') 
    qml.default_config['qiskit.ibmq.ibmqx_token'] = key
    qml.default_config.save(qml.default_config.path)
    backend_config = {'backend': 'qiskit.ibmq',
                    'device': 'ibm_kyoto',
                    'shots': 1000}
else:
    backend_config = {'backend': 'default.qubit'}
model = PennyLaneModel.from_diagrams(all_circuits, probabilities=True, normalize=True, 
                                     backend_config=backend_config)
model.initialise_weights()

# create datasets
train_dataset = Dataset(train_circuits, train_labels, batch_size=BATCH_SIZE)
val_dataset = Dataset(val_circuits, val_labels)

# initialize trainer
trainer = PytorchTrainer(model=model, loss_function=loss, optimizer=optimizer, learning_rate=LEARNING_RATE,
    epochs=EPOCHS, evaluate_functions=evaluate_functions, evaluate_on_train=True, use_tensorboard=False, 
    verbose='text', log_dir=log_dir, seed=SEED)

# train model
trainer.fit(train_dataset, val_dataset)

# visualize results
visualizer(trainer)

# print test set accuracy
pred = model(test_circuits)
labels = torch.tensor(test_labels)
print('Final test accuracy: {}'.format(acc(pred, labels)))
logging.info('Final test accuracy: {}'.format(acc(pred, labels)))