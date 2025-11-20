import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass

        # Store input for backward pass
        self.A = A
        self.A_shape = A.shape

        # Flatten all leading dimensions into one batch dimension
        A_2d = A.reshape(-1, self.W.shape[1])      # (B, in_features)
        self.A_2d = A_2d

        # Linear transform: Z = A W^T + b
        Z_2d = A_2d @ self.W.T + self.b            # (B, out_features)

        # Reshape back to original batch dims with out_features at the end
        Z = Z_2d.reshape(*self.A_shape[:-1], self.W.shape[0])
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass

        # Flatten gradient to match 2D view used in forward
        dLdZ_2d = dLdZ.reshape(-1, self.W.shape[0])   # (B, out_features)

        # Gradients wrt input, weights, and bias
        dLdA_2d = dLdZ_2d @ self.W                    # (B, in_features)
        dLdW = dLdZ_2d.T @ self.A_2d                  # (out_features, in_features)
        dLdb = dLdZ_2d.sum(axis=0)                    # (out_features,)

        # Reshape dLdA back to original input shape
        dLdA = dLdA_2d.reshape(self.A_shape)

        # Store gradients
        self.dLdA = dLdA
        self.dLdW = dLdW
        self.dLdb = dLdb

        # Return gradient of loss wrt input
        return dLdA
