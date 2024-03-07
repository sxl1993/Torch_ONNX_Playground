import torch
from torch import nn

class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.asinh(x)
     
    
from torch.onnx import register_custom_op_symbolic
    
def asinh_symbolic(g, input, *, out=None):
    return g.op("Asinh", input)


register_custom_op_symbolic("aten::asinh", asinh_symbolic, 9)
model = Model()
input = torch.rand(1, 3, 10, 10)

torch.onnx.export(model, input, 'asinh.onnx', input_names=['input'], output_names=['output'], opset_version=9)

torch_output = model(input).detach().numpy()

import onnxruntime
import numpy as np
sess = onnxruntime.InferenceSession(path_or_bytes='asinh.onnx')
ort_output = sess.run(None, {'input': input.numpy()})[0]

assert np.allclose(torch_output, ort_output)