# TFI-matlab
(Preconditioned) Total Field Inversion (TFI) implemented in MATLAB

(C) Matthew Cherukara, 12 March 2024

This is the docstring from mtc_tfi.m:

MTC_TFI Compute total field inversion

    Based on the function "TFI_Linear.py" by C. Boehm, in his wtTFI module

   
Usage:    arr_susc = mtc_tfi(Parameters)

INPUT DATA (REQUIRED):

    Parameters.Field (3D double array) - Unwrapped field map (in ppm)
    
    Parameters.Magnitude (double array) - Magnitude image (can be 3D or 4D)
    
    Parameters.Mask (logical array) - Region of interest mask
    

INPUT PARAMETERS (REQUIRED):

    Parameters.Resolution (vector) - Voxel size in mm
    
    Parameters.Orientation (vector) - Length 3 vector specifying the direction of B0
    
    Parameters.B0 (scalar) - Value of B0 in Tesla
    

INPUT DATA (Optional):

    Parameters.Noise (3D double array) - Noise-map from non-linear complex fitting
    
    Parameters.R2s (3D double array) - R2* map
    
   
INPUT PARAMETERS (Optional, default values will be used if not specified)

    Parameters.MatrixSize (vector) - Should be equal to size(Parameters.Field)
    
    Parameters.Lambda (scalar) - Regularization parameter (default = 1e-5)
    
    Parameters.Precond (scalar) - Binary preconditioner value (default = 30)
    
    Parameters.CPU (boolean) - To run everything on the CPU, this should be 1
    

OUTPUT

    arr_susc (double array) - Susceptibility map
    

DEPENDENCIES (check nested dependencies)

    ak_Dk (Anita's k-space dipole function)
    
   
Things are currently set up to run on the GPU, but it should work on CPU too.


CHANGELOG:

