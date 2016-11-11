"""
Test your network here!

No need to change this code, but feel free to tweak it
to test your network!

Make your changes to Sigmoid#backward in miniflow.py
"""

import numpy as np
from miniflow2 import *

inputs, weights, bias = Input(), Input(), Input()

x = np.array([[-1., -2.], [-1, -2]])
w = np.array([[2., -3], [2., -3]])
b = np.array([-3., -5])
ideal_output = np.array(
    [[1.23394576e-04, 9.82013790e-01],
     [1.23394576e-04, 9.82013790e-01]])

f = Linear(inputs, weights, bias)
g = Sigmoid(f)
cost = MSE(g)

feed_dict = {inputs: x, weights: w, bias: b}
gradients = forward_and_backward(feed_dict, ideal_output, [weights, bias])

"""
You should see a list of gradients on the weights and bias that looks like:
[array([[  6.79488741e-18,  -2.67826355e-12],
       [  1.35897748e-17,  -5.35652711e-12]]), array([ -6.79488741e-18,   2.67826355e-12])]
"""
print(gradients)