class Neuron:
    # takes in a list of inbound neurons, since can have many inputs
    def __init__(self, inbound_neurons=[]):
        # some properties of this Neuron
        # this neuron has a property called "inbound_neurons" which is set equal to the
        # parameter passed in, of the same name
        self.inbound_neurons = inbound_neurons
        # this neuron feeds to a bunch of other neurons: outputs only one value, but
        # feeds it to all of them
        self.outbound_neurons = []
        # value calculated by the neuron itself
        # initialize to None for now
        self.value = None

        # Set this neuron as an outbound neuron on its inputs.
        for n in self.inbound_neurons:
            n.outbound_neurons.append(self)

    # we leave forward() and backward() empty on the main Neuron class
    # because we implement them in subclasses (specific types of Neurons)
    def forward(self):
        """Forward propagation.
        Computes an outbound value based in inbound_neurons
        and stores the value in self.value.
        """
        raise NotImplementedError
        # interesting...should not be "raise NotImplemented", not a value error type
        # throws TypeError instead

    def backward(self):
        """Backward propagation.

        Fill in later.
        """
        raise NotImplementedError


class Input(Neuron):
    # an example of class inheritance: Input is a subclass of Neuron
    def __init__(self):
        # Input neuron has no previous neurons
        # So no need to pass anything into the Neuron instantiator:
        Neuron.__init__(self)

    # Input neuron is the only node that may be passed as an argument to forward()
    # All other neurons should get their previous values from self.inbound_neurons
    # example:
    # val0 = self.inbound_neurons[0].value

    def forward(self, value=None):
        # Overwrite value if one is passed in
        if value:
            self.value = value

    # Note that input doesn't calculate anything
    # Just holds a value which is explicitly set, or given in forward(value=whatever)


class Add(Neuron):
    # another subclass of Neuron, this one adds
    def __init__(self, *inputs):
        input_list = [input for input in inputs]
        Neuron.__init__(self, input_list)

    def forward(self):
        sum_value = 0
        # add up all the values from inbound_neurons
        for inbound in self.inbound_neurons:
            sum_value += inbound.value
        # set this Add Neuron's own value to sum_value
        self.value = sum_value

class Mul(Neuron):
    # another subclass of Neuron, this one multiplies
    def __init__(self, *inputs):
        input_list = [input for input in inputs]
        Neuron.__init__(self, input_list)

    def forward(self):
        product_value = 1
        # add up all the values from inbound_neurons
        for inbound in self.inbound_neurons:
            product_value *= inbound.value
        # set this Add Neuron's own value to sum_value
        self.value = product_value

class Linear(Neuron):
    # another subclass of Neuron, this one does dot product and adds a bias
    def __init__(self, inputs, weights, bias):
        Neuron.__init__(self, inputs)
        self.weights = weights
        self.bias = bias

    def forward(self):
        sumproduct = 0
        i = 0
        while i < len(self.weights):
            sumproduct += self.weights[i].value * self.inbound_neurons[i].value
            i += 1
        sumproduct += self.bias.value
        self.value = sumproduct


## TOPOLOGICAL SORT
# what is a topological sort?
# there is an order of operations
# given that neurons take inputs from other neurons
# we must flatten the network and ensure all inputs are available before doing the calculation

# topological_sort() returns a sorted list of Neurons
# all calculations can run in series
# input is feed_dict, a dictionary of input values

# example:

# instantiate the Input Neurons x and y
# x, y = Input(), Input()
# return a list of sorted Neurons:
# sorted_neurons = topological_sort(feed_dict={x: 10, y: 20})

# copied from class
def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_neurons = [n for n in feed_dict.keys()]

    G = {}
    neurons = [n for n in input_neurons]
    while len(neurons) > 0:
        n = neurons.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_neurons:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            neurons.append(m)

    L = []
    S = set(input_neurons)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_neurons:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


## FORWARD PASS
# this actually runs the network and outputs a value
def forward_pass(output_neuron, sorted_neurons):
    """
    Performs forward pass through list of sorted Neurons.
    ( basically just loops through everything in the sorted list and runs forward(),
     then grabs the value of output_neuron and returns it )
    Arguments:
        'output_neuron: The final output neuron of graph (no outgoing edges)
        'sorted_neurons': returned by topological_sort()
    Returns the output_neuron's value.
    """

    for n in sorted_neurons:
        n.forward()

    return output_neuron.value


def main():
    neuron = Neuron()
    neuron.forward()


if __name__ == "__main__":
    main()
