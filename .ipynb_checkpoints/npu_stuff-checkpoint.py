import npu
import numpy as np
from npu.build.appbuilder import AppBuilder
from npu.build.kernel import Kernel
from npu.runtime import AppRunner
from npu.utils import OpenCVImageReader, image_plot
from PIL import Image
import pylab as plt

# Making and building the application
class Application(AppBuilder):

    def __init__(self, kernel:Kernel):
        self.kernel = kernel
        super().__init__()

    def callgraph(self, x_in: np.ndarray, x_in2: np.ndarray, x_out: np.ndarray) -> None:
        rows = x_in.shape[0]
        row_len = x_in.shape[1]
        for row in range(rows):
            kernel_output = self.kernel(x_in[row], row_len, x_in2[row])
            x_out[row] = kernel_output


def buildApp(M, P, W, kernels):
    dt = np.float32
    app_builder = Application(kernels[0])
    
    inputs = np.zeros(shape=(1, M*P), dtype=dt)
    outputs = np.zeros(shape=(1, M*P), dtype=dt)
    
    app_builder.build(inputs, inputs, outputs)

    app = AppRunner('Application.xclbin')
    return app

def runApp(app, input_data, input_data2, output_data, i):
    dt = np.float32
    # if i <= 5:
    #     plt.plot(input_data)
    #     plt.savefig(f'./plots/input{i}.png')
    #     plt.clf()
    
    inputs = app.allocate(shape=input_data.shape, dtype=dt)
    inputs2 = app.allocate(shape=input_data.shape, dtype=dt)
    outputs = app.allocate(shape=output_data.shape, dtype=dt)
    
    inputs[:] = input_data        
    inputs2[:] = input_data2
    inputs.sync_to_npu()
    inputs2.sync_to_npu()
    
    app.call(inputs, inputs2, outputs)
    
    outputs.sync_from_npu()
    # if i <= 5:
    #     plt.plot(outputs)
    #     plt.savefig(f'./plots/gg{i}.png')
    #     plt.clf()
    
    return outputs[0]
    