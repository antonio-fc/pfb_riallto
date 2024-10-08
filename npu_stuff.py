import npu
import numpy as np
from npu.build.appbuilder import AppBuilder
from npu.build.kernel import Kernel
from npu.runtime import AppRunner
from npu.utils import OpenCVImageReader, image_plot
from PIL import Image

# Making and building the application
class Application(AppBuilder):

    def __init__(self, kernel:Kernel):
        self.kernel = kernel
        super().__init__()

    def callgraph(self, x_in: np.ndarray, x_out: np.ndarray) -> None:
        rows = x_in.shape[0]
        row_len = x_in.shape[1]
        for row in range(rows):
            kernel_output = self.kernel(x_in[row], row_len)
            x_out[row] = kernel_output


def buildApp(M, P, W, kernels):
    app_builder = Application(kernels[0])
    
    inputs = np.zeros(shape=(1, M*P), dtype=np.uint16)
    outputs = np.zeros(shape=(1, M*P), dtype=np.uint16)
    
    app_builder.build(inputs, outputs)

    app = AppRunner('Application.xclbin')
    return app

def runApp(app, input_data, output_data):
    
    inputs = app.allocate(shape=input_data.shape, dtype=np.float16)
    outputs = app.allocate(shape=output_data.shape, dtype=np.float16)
    inputs[:] = input_data
    inputs.sync_to_npu()
    app.call(inputs, outputs)
    outputs.sync_from_npu()
    return_data = np.zeros(shape=output_data.shape, dtype=np.float16)
    return_data[:] = outputs
    
    del app
    
    return return_data
    