import numpy as np
import matlab.engine

class connect_matlab(object):

    def __enter__(self):
        print("Connecting to MATLAB... ", end="", flush=True)
        self.eng = matlab.engine.connect_matlab()
        print("done")
        return self.eng

    def __exit__(self, type, value, traceback):
        print("Disconnecting from MATLAB... ", end="", flush=True)
        self.eng.quit()
        print("done")
        
def ndarray_to_matlab(a):
    if a.dtype in [np.complex64, np.complex128]:
        is_complex = True
    else:
        is_complex = False

    return matlab.double(a.tolist(), is_complex=is_complex)
    
    
