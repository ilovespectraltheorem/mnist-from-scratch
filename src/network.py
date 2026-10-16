"""A network is a list of layers."""
import numpy as np
from src.losses import softmax


class Sequential:
    def __init__(self, layers):
        self.layers = list(layers)

    def train(self):
        """Training mode: dropout on."""
        for layer in self.layers:
            layer.training = True

    def eval(self):
        """Evaluation mode: dropout off."""
        for layer in self.layers:
            layer.training = False

    def parameters(self):
        """Yields (key, param_array, grad_array) for every parameter, e.g. ("0.W", W, dW).
        The key is "<layer index>.<name>", so it is unique within the network."""
        for i, layer in enumerate(self.layers):
            for name, value in layer.params.items():
                yield f"{i}.{name}", value, layer.grads.get(name)

    def num_params(self):
        return sum(p.size for _, p, _ in self.parameters())

    def state_dict(self):
        """A copy of every parameter (used to keep the best model for early stopping)."""
        return {key: p.copy() for key, p, _ in self.parameters()}

    def load_state_dict(self, state):
        for key, p, _ in self.parameters():
            p[...] = state[key]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        """Backpropagate through every layer in reverse order and return the
        gradient with respect to the network's input."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def logits(self, X, batch_size=1000):
        """Inference in evaluation mode, in chunks of batch_size to bound memory.
        Each layer's previous mode is restored afterwards."""
        was_training = [layer.training for layer in self.layers]
        self.eval()
        out = [self.forward(X[i:i + batch_size]) for i in range(0, len(X), batch_size)]
        for layer, flag in zip(self.layers, was_training):
            layer.training = flag
        return np.concatenate(out, axis=0)

    def predict_proba(self, X, batch_size=1000):
        """Class probabilities, shape (N, 10)."""
        return softmax(self.logits(X, batch_size))

    def predict(self, X, batch_size=1000):
        """Predicted class for each row, shape (N,)."""
        return np.argmax(self.logits(X, batch_size), axis=1)
