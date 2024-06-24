import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class DenseLayer:
    
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.output = np.dot(inputs, self.weights) + self.biases
        
class ActivationReLU:
    
    def forward(self,inputs: np.ndarray) -> None:
       self.output = np.maximum(0,inputs) 
        
class ActivationSoftmax:
    
     def forward(self, inputs: np.ndarray) -> None:
         exp_max_input = np.exp(inputs - np.max(inputs, axis=1,keepdims=True))
         sum_input = np.sum(exp_max_input, axis=1, keepdims=True)
         self.output = exp_max_input/sum_input

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

print(loss.calculate(activation2.output, y ))