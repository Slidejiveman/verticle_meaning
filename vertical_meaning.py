from jp_grammar_model import * # grammar types
from jp_grammar_utilities import diagramizer, quantizer, visualizer
from lambeq import BinaryCrossEntropyLoss, Dataset, IQPAnsatz, NumpyModel, QuantumTrainer, SPSAOptimizer, TketModel, UnifyCodomainRewriter
from pytket.extensions.qiskit import AerBackend
import jax.numpy as jnp
import logging
import numpy as np

## Setup logging
logging.basicConfig(filename='./logs/quantum/app/app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

## Open datasets and collect sentences
def read_data(filename):
    labels, sentences = [], []
    with open(filename) as file:
        for line in file:
            t = int(line[0])
            labels.append([t, 1-t]) # returns an array with the correct label in first place
            sentences.append(line[1:].strip())
    return labels, sentences

## Define accuracy metric
def acc(y_hat, y):
    if isinstance(y_hat, jnp.ndarray):
        y_hat = np.asarray(y_hat)
    if isinstance(y, jnp.ndarray):
        y = np.asarray(y)
    accuracy = np.sum(np.round(y_hat) == y) / len(y) / 2
    return accuracy

## define utilities and hyperparameters
BATCH_SIZE = 2
EPOCHS = 100
SEED = 2
optimizer = SPSAOptimizer
alpha = {'a': 0.05} 
c = {'c': 0.06}
A = {'A': 0.001 * EPOCHS}
eval_metrics = {"acc": acc}
optim_hyperparams = alpha | c | A
is_fast_test = True
bce = BinaryCrossEntropyLoss() 
log_dir='./logs/quantum'
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
if is_fast_test:
    model = NumpyModel.from_diagrams(all_circuits, use_jit=True)
else: 
    backend = AerBackend()
    backend_config = {
        'backend': backend,
        'compilation': backend.default_compilation_pass(2),
        'shots': 8192
    }
    model = TketModel.from_diagrams(all_circuits, backend_config=backend_config)

# initialize trainer
trainer = QuantumTrainer(model, loss_function=bce, epochs=EPOCHS, optimizer=optimizer,
    optim_hyperparams=optim_hyperparams, evaluate_functions=eval_metrics,
    evaluate_on_train=True, verbose='text', log_dir=log_dir, seed=SEED)

# create datasets
train_dataset = Dataset(train_circuits, train_labels, batch_size=BATCH_SIZE)
val_dataset = Dataset(val_circuits, val_labels, shuffle=False)

# train model
trainer.fit(train_dataset, val_dataset, early_stopping_interval=10)

# visualize results 
visualizer(trainer)

# print test accuracy
model.load(trainer.log_dir + '/best_model.lt')
test_acc = acc(model(test_circuits), test_labels)
print('Test set accuracy:', test_acc.item())
logging.info('Test set accuracy:', test_acc.item())