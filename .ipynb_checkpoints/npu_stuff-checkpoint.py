import npu
import numpy as np
import pylab as plt
from npu.build.appbuilder import AppBuilder
from npu.build.kernel import Kernel
from npu.runtime import AppRunner
from npu.build.mtkernel import MTSplit, MTConcat
from npu.build.mtkernel import MTPassThrough

# Making and building the application
class Application(AppBuilder):
    
    def __init__(self, kernel):
        self.kernel = kernel
        self.mtbuffer_in = MTPassThrough()
        super().__init__()

    def callgraph(self, x_in:np.ndarray, x_in2:np.ndarray, x_out:np.ndarray):
        rows = x_in.shape[0]
        bytes_per_row = x_in.shape[1]
        for row in range(rows):
            mtpin = self.mtbuffer_in(x_in[row])
            kernel_output = self.kernel(mtpin, x_in[row], bytes_per_row)
            x_out[row] = kernel_output

def buildApp(kernels, M, P, dt):
    # Making the app
    app_builder = Application(kernel=kernels[0])

    # Building the app
    input_form = np.zeros(shape=(1, M*P), dtype=dt)
    input_form2 = np.zeros(shape=(1, M*P), dtype=dt)
    output_form = np.zeros(shape=(1, M*P), dtype=dt)
    
    app_builder.build(input_form, input_form2, output_form)

    app = AppRunner('Application.xclbin')
    
    return app

def runApp(app, M, P, data, coeffs, dt):
    # Making app runner and running app
    input_data = app.allocate(shape=(1, M*P), dtype=dt)
    input_data2 = app.allocate(shape=(1, M*P), dtype=dt)
    output_data = app.allocate(shape=(1, M*P), dtype=dt)

    input_data[:] = data
    input_data.sync_to_npu()
    input_data2[:] = coeffs
    input_data2.sync_to_npu()
    
    app.call(input_data, input_data2, output_data)
    
    output_data.sync_from_npu()

    return output_data