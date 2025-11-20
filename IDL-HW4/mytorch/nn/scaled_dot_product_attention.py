import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # Initialize your softmax layer
        # What dimension should you pass to the softmax constructor?
        self.eps = 1e10 # DO NOT MODIFY
        # Perform softmax on the last dimension S
        self.softmax = Softmax(dim=-1)
        
    
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape where 1/True indicates a position to ignore
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        # Save input for backward use
        self.Q = Q
        self.K = K
        self.V = V
        self.d_k = Q.shape[-1]

        # (N, ..., H, L, E) @ (N, ..., H, E, S) -> (N, ..., H, L, S)
        scaled_dot_product = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(self.d_k)

        # Apply mask before softmax if provided
        if mask is not None:
            # Subtracting a huge number from the position of True/1 is equivalent to -inf
            scaled_dot_product = scaled_dot_product - self.eps * mask

        # Compute attention scores: softmax over S dim
        self.attention_scores = self.softmax.forward(scaled_dot_product)

        # (N, ..., H, L, S) @ (N, ..., H, S, Ev) -> (N, ..., H, L, Ev) 
        output = np.matmul(self.attention_scores, V)

        return output
    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        A = self.attention_scores
        Q = self.Q
        K = self.K
        V = self.V

        # dV: A^T @ d_output  -> (N, ..., H, S, Ev)
        d_V = np.matmul(np.swapaxes(A, -2, -1), d_output)
        
        # dA: d_output @ V^T  -> (N, ..., H, L, S)
        d_attention_scores = np.matmul(d_output, np.swapaxes(V, -1, -2))

        # The gradient with respect to scaled scores is obtained through softmax.
        d_scaled_dot_product = self.softmax.backward(d_attention_scores)
        
        # Consider scaling by dividing by sqrt(d_k): scores = (QK^T)/sqrt(d_k)
        d_scaled_dot_product = d_scaled_dot_product / np.sqrt(self.d_k)
        
        # dQ: d_scores @ K -> (N, ..., H, L, E)
        d_Q = np.matmul(d_scaled_dot_product, K)

        # dK: d_scores^T @ Q -> (N, ..., H, S, E)
        d_K = np.matmul(np.swapaxes(d_scaled_dot_product, -2, -1), Q)
        
        return d_Q, d_K, d_V
