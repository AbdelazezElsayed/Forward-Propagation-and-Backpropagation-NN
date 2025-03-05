import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class SimpleNN:
    def __init__(self):
        self.w1 = random.uniform(-0.5, 0.5)
        self.w2 = random.uniform(-0.5, 0.5)
        self.w3 = random.uniform(-0.5, 0.5)
        self.w4 = random.uniform(-0.5, 0.5)
        self.w5 = random.uniform(-0.5, 0.5)
        self.w6 = random.uniform(-0.5, 0.5)
        self.w7 = random.uniform(-0.5, 0.5)
        self.w8 = random.uniform(-0.5, 0.5)
        self.b1 = 0.5
        self.b2 = 0.7
        self.learning_rate = 0.1

    def forward(self, x1, x2):
        self.x1, self.x2 = x1, x2
        self.z1 = self.w1 * x1 + self.w2 * x2 + self.b1
        self.h1 = sigmoid(self.z1)
        self.z2 = self.w3 * x1 + self.w4 * x2 + self.b1
        self.h2 = sigmoid(self.z2)
        self.z3 = self.w5 * self.h1 + self.w6 * self.h2 + self.b2
        self.o1 = sigmoid(self.z3)
        self.z4 = self.w7 * self.h1 + self.w8 * self.h2 + self.b2
        self.o2 = sigmoid(self.z4)
        return self.o1, self.o2

    def backward(self, target1, target2):
        error1 = self.o1 - target1
        delta_o1 = error1 * sigmoid_derivative(self.z3)
        error2 = self.o2 - target2
        delta_o2 = error2 * sigmoid_derivative(self.z4)

        delta_w5 = delta_o1 * self.h1
        delta_w6 = delta_o1 * self.h2
        delta_w7 = delta_o2 * self.h1
        delta_w8 = delta_o2 * self.h2
        delta_b2 = delta_o1 + delta_o2

        delta_h1 = delta_o1 * self.w5 + delta_o2 * self.w7
        delta_h1 = delta_h1 * sigmoid_derivative(self.z1)
        delta_h2 = delta_o1 * self.w6 + delta_o2 * self.w8
        delta_h2 = delta_h2 * sigmoid_derivative(self.z2)

        delta_w1 = delta_h1 * self.x1
        delta_w2 = delta_h1 * self.x2
        delta_w3 = delta_h2 * self.x1
        delta_w4 = delta_h2 * self.x2
        delta_b1 = delta_h1 + delta_h2

        self.w1 -= self.learning_rate * delta_w1
        self.w2 -= self.learning_rate * delta_w2
        self.w3 -= self.learning_rate * delta_w3
        self.w4 -= self.learning_rate * delta_w4
        self.w5 -= self.learning_rate * delta_w5
        self.w6 -= self.learning_rate * delta_w6
        self.w7 -= self.learning_rate * delta_w7
        self.w8 -= self.learning_rate * delta_w8
        self.b1 -= self.learning_rate * delta_b1
        self.b2 -= self.learning_rate * delta_b2

    def train(self, x1, x2, target1, target2, epochs):
        for i in range(epochs):
            o1, o2 = self.forward(x1, x2)
            self.backward(target1, target2)
            error1 = o1 - target1
            error2 = o2 - target2
            print(f"Epoch {i + 1}, Output: ({o1}, {o2}), Error: ({error1}, {error2})")

nn = SimpleNN()
x1, x2 = 0.1, 0.2
target1, target2 = 0.5, 0.3
nn.train(x1, x2, target1, target2, 10)

print(f"Weights: {nn.w1}, {nn.w2}, {nn.w3}, {nn.w4}, {nn.w5}, {nn.w6}, {nn.w7}, {nn.w8}")
print(f"Biases: {nn.b1}, {nn.b2}")
print(f"Input: {x1}, {x2}")
print(f"Final Output: {nn.forward(x1, x2)}")
