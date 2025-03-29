import numpy as np
import torch
from torch import nn

def test_linear_layer():
    # Parameters for the test
    batch_size = 1024
    input_features = 1024
    output_features = 1024
    
    # Random input, weight, and bias initialization
    X = np.random.randn(batch_size, input_features).astype(np.float32)
    W = np.random.randn(input_features, output_features).astype(np.float32)
    B = np.random.randn(output_features).astype(np.float32)
    
    # Define PyTorch linear layer and set weights and biases
    linear_layer = nn.Linear(input_features, output_features, bias=True)
    with torch.no_grad():
        # PyTorch stores weights as [out_features, in_features]
        linear_layer.weight.copy_(torch.from_numpy(W.T))
        linear_layer.bias.copy_(torch.from_numpy(B))
    
    # Compute reference output
    ref = linear_layer(torch.from_numpy(X)).detach().numpy()
    
    # Write input data to files for C++ implementation
    # input0.data: X flattened (batch_size * input_features floats)
    with open("input0.data", "w") as f:
        for val in X.flatten():
            f.write(f"{val}\n")
    
    # input1.data: W flattened (input_features * output_features floats)
    with open("input1.data", "w") as f:
        for val in W.flatten():
            f.write(f"{val}\n")
    
    # input2.data: B (output_features floats)
    with open("input2.data", "w") as f:
        for val in B:
            f.write(f"{val}\n")
    
    # Write PyTorch output to output.data
    with open("output.data", "w") as f:
        for val in ref.flatten():
            f.write(f"{val}\n")
    
    print("Files generated:")
    print("- input0.data: Input data")
    print("- input1.data: Weights")
    print("- input2.data: Bias")
    print("- output.data: PyTorch reference output")
    print("Ready for C++ implementation testing.")

if __name__ == "__main__":
    test_linear_layer()