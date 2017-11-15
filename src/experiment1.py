from network import Network
import numpy as np
inputs = np.array([[1]]).T
input_index = np.array([[0]])
output = np.array([[0, 1, 0, 0]]).T
output_index = np.array([[50, 51, 52, 53]])

net = Network(100)
# Train
for i in range(1000):
    net.update()
    net.input(inputs, input_index)
    net.display()
    net.update()
    print(net.output(output, output_index))

# Test
for i in range(1000):
    net.update()
    net.display()
    net.input(inputs, input_index)
    net.update()
