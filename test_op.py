import onnx
import numpy as np
from onnx.backend.test.case.node import expect

node = onnx.helper.make_node(
    'Asinh',
    inputs=['x'],
    outputs=['y']
)

x = np.array([-1, 0, 1]).astype(np.float32)
y = np.arcsinh(x)

expect(node, inputs=[x], outputs=[y], name="test_asinh_example")

