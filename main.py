import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class DenseLayer:
    
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    
    def forward(self, inputs: np.ndarray) -> None:
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
    
    def backward(self, dvalues: np.ndarray) -> None:
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        
class ActivationReLU:
    
    def forward(self, inputs: np.ndarray) -> None:
       self.output = np.maximum(0,inputs)
       self.inputs = inputs

    def backward(self, dvalues: np.ndarray) -> None:
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        
class ActivationSoftmax:
    
    def forward(self, inputs: np.ndarray) -> None:
        self.inputs = inputs
        exp_max_input = np.exp(inputs - np.max(inputs, axis=1,keepdims=True))
        sum_input = np.sum(exp_max_input, axis=1, keepdims=True)
        self.output = exp_max_input / sum_input

    def backward(self, dvalues: np.ndarray) -> None:
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, -1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss:
    
    def calculate(self, output: np.ndarray, y: np.ndarray) -> float:
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)

class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        n_samples = len(y_pred)
        y_pred_clip = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        if len(y_true.shape) == 1:
            confidences = y_pred_clip[range(n_samples),y_true]
        elif len(y_true.shape) == 2:
            confidences = np.sum(y_pred_clip*y_true, axis=1)
        
        return -np.log(confidences)

    def backward(self, dvalues: np.ndarray, y_true: np.ndarray) -> None:
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples
class ActivationSoftmaxCrossEntropy():
    def __init__(self):
        self.activation = ActivationSoftmax()
        self.loss = CategoricalCrossEntropy()

    def forward(self, inputs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues: np.ndarray, y_true: np.ndarray) -> None:
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    
    predictions = np.argmax(y_pred, axis=1)
    
    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)
    
    return np.mean(predictions==y_true)
    
X, y = spiral_data(samples=100,classes=3)

dense1 = DenseLayer(2,3)
activation1 = ActivationReLU()
dense2 = DenseLayer(3,3)
activation2 = ActivationSoftmax()
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output)

loss = CategoricalCrossEntropy()

print(loss.calculate(activation2.output, y))

print(accuracy(activation2.output, y))
