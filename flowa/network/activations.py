import numpy as np

class Sigmoid:
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        return x * (1 - x)

class Tanh:
    def forward(self, x):
        return np.tanh(x)

    def backward(self, x):
        return 1 - x ** 2

class ReLU:
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return 1 * (x > 0)

class Softmax:
    def forward(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def backward(self, x):
        return x * (1 - x)

class _LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        return np.maximum(self.alpha * x, x)

    def backward(self, x):
        return 1 * (x > 0)

class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.leaky_relu = _LeakyReLU(alpha)

    def forward(self, x):
        return self.leaky_relu.forward(x)

    def backward(self, x):
        return self.leaky_relu.backward(x)

class _ELU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def backward(self, x):
        return 1 * (x > 0)

class ELU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.elu = _ELU(alpha)

    def forward(self, x):
        return self.elu.forward(x)

    def backward(self, x):
        return self.elu.backward(x)

class Swish:
    def forward(self, x):
        return x * np.tanh(np.sqrt(2 / np.pi) * x)

    def backward(self, x):
        return x * (1 + np.tanh(np.sqrt(2 / np.pi) * x) ** 2)

class Gaussion:
    def forward(self, x):
        return np.exp(-x ** 2)

    def backward(self, x):
        return -2 * x

class Identity:
    def forward(self, x):
        return x

    def backward(self, x):
        return 1

class BinaryStep:
    def forward(self, x):
        return np.where(x > 0, 1, 0)

    def backward(self, x):
        return 1 * (x > 0)

class _PReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        return np.maximum(self.alpha * x, x)

    def backward(self, x):
        return 1 * (x > 0)

class PReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.prelu = _PReLU(alpha)

    def forward(self, x):
        return self.prelu.forward(x)

    def backward(self, x):
        return self.prelu.backward(x)

class Exponential:
    def forward(self, x):
        return np.exp(x)

    def backward(self, x):
        return np.exp(x)

class Softplus:
    def forward(self, x):
        return np.log(1 + np.exp(x))

    def backward(self, x):
        return 1 / (1 + np.exp(-x))

class Softsign:
    def forward(self, x):
        return x / (1 + np.abs(x))

    def backward(self, x):
        return 1 / (1 + np.abs(x))

class BentIdentity:
    def forward(self, x):
        return x * np.sign(x)

    def backward(self, x):
        return 1

class ArcTan:
    def forward(self, x):
        return np.arctan(x)

    def backward(self, x):
        return 1 / (1 + x ** 2)

class SiLU:
    def forward(self, x):
        return x * (1 / (1 + np.exp(-x)))

    def backward(self, x):
        return x * (1 + np.exp(x))

class Mish:
    def forward(self, x):
        return x * np.tanh(np.sqrt(2 / np.pi) * x)

    def backward(self, x):
        return x * (1 + np.tanh(np.sqrt(2 / np.pi) * x) ** 2)

class HardSigmoid:
    def forward(self, x):
        return np.where(x > 0, 1, 0)

    def backward(self, x):
        return 1 * (x > 0)

class HardTanh:
    def forward(self, x):
        return np.where(x > 0, x, 0)

    def backward(self, x):
        return 1 * (x > 0)

class SoftExponential:
    def forward(self, x):
        return np.exp(x) / (1 + np.exp(x))

    def backward(self, x):
        return np.exp(x) / (1 + np.exp(x)) ** 2

class ISRU:
    def forward(self, x):
        return np.where(x > 0, x, 0)

    def backward(self, x):
        return 1 * (x > 0)

class Sine:
    def forward(self, x):
        return np.sin(x)

    def backward(self, x):
        return np.cos(x)

class Cosine:
    def forward(self, x):
        return np.cos(x)

    def backward(self, x):
        return -np.sin(x)

class SQNL:
    def forward(self, x):
        return np.where(x > 0, x, 0)

    def backward(self, x):
        return 1 * (x > 0)

class SoftClipping:
    def forward(self, x):
        return np.where(x > 0, x, 0)

    def backward(self, x):
        return 1 * (x > 0)

class BentIdentity2:
    def forward(self, x):
        return x * np.sign(x)

    def backward(self, x):
        return 1

class LogLog:
    def forward(self, x):
        return np.log(np.log(x))

    def backward(self, x):
        return 1 / (np.log(x) ** 2)

class GELU:
    def forward(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def backward(self, x):
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

class Softmin:
    def forward(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def backward(self, x):
        return np.exp(x) / np.sum(np.exp(x))
