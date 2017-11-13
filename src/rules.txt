Properties
Randomly initialize connections
Circular arrangement of neurons with connections from anywhere across the circle
Input and output lie on the border of the cortex
When randomly initialized, random inputs and outputs will cause the network to light up - important to prevent the connections from all dying

Rules:
Hebbian Learning: If an axon tries to trigger a neuron that is already activated, strength the connection
Oja's Rule: change in delta weight = alpha*y*(x-y*w)
where y = output, x = input, w = weight
This is a normalized Hebb's rules so the weights don't go to infinity
Overly activated neurons kill themselves
Any input associated with a firing of the neuron gets a stronger weight
During testing, neurons that recently fired during a wrong answer get weakened
If the input fires a neuron that has already fires, strengthen the previous axons

Variants:
Neurons can only have one axon
Different neuron models

Tests:
Sliding window around cortex to make it predict the next frame
Repeating patterns
