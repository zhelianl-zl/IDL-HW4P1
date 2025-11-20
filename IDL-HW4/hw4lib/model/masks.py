import torch

''' 
TODO: Implement this function.

Specification:
- Function should create a padding mask that identifies padded positions in the input
- Mask should be a boolean tensor of shape (N, T) where:
  * N = batch size from padded_input
  * T = sequence length from padded_input
- True values indicate padding positions that should be masked
- False values indicate valid positions that should not be masked
- Padding is assumed to be on the right side of sequences
- Each sequence in the batch may have different valid lengths
- Mask should be on same device as input tensor
'''
def PadMask(padded_input, input_lengths):
    """ 
    Create a mask to identify non-padding positions. 
    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
        input_lengths: The actual lengths of each sequence before padding, shape (N,).
    Returns:
        A boolean mask tensor with shape (N, T), where: 
            - padding positions are marked with True 
            - non-padding positions are marked with False.
    """
    # TODO: Implement PadMask
    """Create a mask to identify padding positions."""
    # padded_input: (N, T, ...) or (N, T)
    # input_lengths: (N,)
    device = padded_input.device
    N, T = padded_input.shape[:2]

    # location index：0..T-1  -> (N, T)
    positions = torch.arange(T, device=device).unsqueeze(0).expand(N, T)
    lengths   = input_lengths.to(device).unsqueeze(1).expand(N, T)

    # True = padding (after the actual length)
    mask = positions >= lengths          # (N, T), bool
    return mask
    

''' 
TODO: Implement this function.

Specification:
- Function should create a causal mask for self-attention
- Mask should be a boolean tensor of shape (T, T) where T is sequence length
- True values indicate positions that should not attend to each other
- False values indicate positions that can attend to each other
- Causal means each position can only attend to itself and previous positions
- Mask should be on same device as input tensor
- Mask should be upper triangular (excluding diagonal)
'''
def CausalMask(padded_input):
    """ 
    Create a mask to identify non-causal positions. 
    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
    
    Returns:
        A boolean mask tensor with shape (T, T), where: 
            - non-causal positions (don't attend to) are marked with True 
            - causal positions (can attend to) are marked with False.
    """
    # TODO: Implement CausalMask
    """Create a causal mask over sequence length."""
    # padded_input: (N, T, ...) or (N, T)
    device = padded_input.device
    T = padded_input.shape[1]

    # 上三角（不含对角线）为 True = 不能看未来
    ones = torch.ones((T, T), dtype=torch.bool, device=device)
    mask = torch.triu(ones, diagonal=1)  # (T, T)
    return mask

