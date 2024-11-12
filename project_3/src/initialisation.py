import numpy as np

def initialize(network_shape: list[int], method: str = "Standard") -> list:
    """
    Initialize the weights and biases for a neural network.

    Parameters:
    -----------
    network_shape : list[int]
        A list containing the number of neurons in each layer of the network.
        The first element is the input layer size, and the last element is the output layer size.
    method : str
        The initialization method to use. Options are:
                "Standard" - Standard Normal initialization,
                "Xavier" - Xavier/Glorot initialization,
                "He" - He initialization.
                Default is "Standard".

    Returns:
    --------
    list
        A list of tuples containing the weight matrices and bias vectors for each layer in the network.
        Each tuple is in the form (W, b), where W is the weight matrix and b is the bias vector for a layer.

    Raises:
    -------
    ValueError : If an unsupported initialization method is specified.
    """
    np.random.seed(4155)

    def standard_normal(input: int, output: int) -> tuple[int]:
        b = np.random.randn(output)
        W = np.random.randn(input, output)
        return W, b

    def xavier(input: int, output: int) -> tuple[int]:
        """
        This method initializes weights as described in:
            Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks.
        Retrieved from: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        """
        b = np.zeros(output)
        limit = np.sqrt(6 / float(input + output))
        W = np.random.uniform(low=-limit, high=limit, size=(input, output))
        return W, b

    def he(input: int, output: int) -> tuple[int]:
        """
        This method initializes weights as described in:
            He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving Deep into Rectifiers: Surpassing Human-Level
        Retrived from: https://openaccess.thecvf.com/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
        """
        b = np.zeros(output)
        limit = np.sqrt(2 / float(input))
        W = np.random.normal(0.0, limit, size=(input, output))
        return W, b

    match method:
        case "Standard":
            function = standard_normal
        case "Xavier":
            function = xavier
        case "He":
            function = he
        case _:
            raise ValueError("Unsupported initialization method")

    layers = []
    i_size = network_shape[0]

    for layer_output_size in network_shape[1:]:
        W, b = function(i_size, layer_output_size)
        layers.append((W, b))
        i_size = layer_output_size

    return layers