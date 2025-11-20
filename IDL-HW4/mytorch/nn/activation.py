import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        axis = self.dim if self.dim >= 0 else self.dim + Z.ndim

        Z_max = np.max(Z, axis=axis, keepdims=True)
        Z_shift = Z - Z_max
        expZ = np.exp(Z_shift)
        sum_expZ = np.sum(expZ, axis=axis, keepdims=True)
        A = expZ / sum_expZ

        self.A = A
        return A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # Get the shape of the input
        shape = self.A.shape
        axis = self.dim if self.dim >= 0 else self.dim + len(shape)

        A_moved = np.moveaxis(self.A, axis, -1)
        dLdA_moved = np.moveaxis(dLdA, axis, -1)

        C = A_moved.shape[-1]

        A_2d = A_moved.reshape(-1, C)      # (B, C)
        dLdA_2d = dLdA_moved.reshape(-1, C)

        dot = np.sum(dLdA_2d * A_2d, axis=1, keepdims=True)  # (B, 1)
        dLdZ_2d = A_2d * (dLdA_2d - dot)                     # (B, C)

        dLdZ_moved = dLdZ_2d.reshape(A_moved.shape)

        dLdZ = np.moveaxis(dLdZ_moved, -1, axis)

        return dLdZ


 

    