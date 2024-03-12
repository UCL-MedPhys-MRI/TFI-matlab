function arr_susc = mtc_tfi(Parameters)
% MTC_TFI Compute total field inversion
%   Based on the function "TFI_Linear.py" by C. Boehm, in his wtTFI module
%
% Usage:    arr_susc = mtc_tfi(Parameters)
%
% INPUT DATA (REQUIRED):
%
%   Parameters.Field (3D double array) - Unwrapped field map (in ppm)
%   Parameters.Magnitude (double array) - Magnitude image (can be 3D or 4D)
%   Parameters.Mask (logical array) - Region of interest mask
%
% INPUT PARAMETERS (REQUIRED):
%
%   Parameters.Resolution (vector) - Voxel size in mm
%   Parameters.Orientation (vector) - Length 3 vector specifying the direction of B0
%   Parameters.B0 (scalar) - Value of B0 in Tesla
%
% INPUT DATA (Optional):
%
%   Parameters.Noise (3D double array) - Noise-map from non-linear complex fitting
%   Parameters.R2s (3D double array) - R2* map
%   
% INPUT PARAMETERS (Optional, default values will be used if not specified)
%
%   Parameters.MatrixSize (vector) - Should be equal to size(Parameters.Field)
%   Parameters.Lambda (scalar) - Regularization parameter (default = 1e-5)
%   Parameters.Precond (scalar) - Binary preconditioner value (default = 30)
%   Parameters.CPU (boolean) - To run everything on the CPU, this should be 1
%
% OUTPUT
%  
%   arr_susc (double array) - Susceptibility map
%
% DEPENDENCIES (check nested dependencies)
%
%   ak_Dk (Anita's k-space dipole function)
%   
% Things are currently set up to run on the GPU. I'll add an option to force CPU
% computation if necessary.
% 
%
%       Copyright (C) University College London, 2024
%
%
% Created by MT Cherukara, March 2024
%
% CHANGELOG:


%% CHECK INPUTS

% Data
if isfield(Parameters,'Field')
    arr_field = Parameters.Field;
else
    error('You must provide a FIELD map!');
end

if isfield(Parameters,'Magnitude')
    arr_magn = Parameters.Magnitude;
else
    error('You must provide a MAGNITUDE image!');
end

if isfield(Parameters,'Mask')
    arr_mask = Parameters.Mask == 1;
else
    error('You must provide a binary MASK!');
end

if ~isfield(Parameters,'Resolution')
    warning('Resolution not specified, defaulting to [1, 1, 1]')
    Parameters.Resolution = [1,1,1];
end

if ~isfield(Parameters,'Orientation')
    warning('Orientation not specified, defaulting to [0, 0, 1]')
    Parameters.Orientation = [0,0,1];
end

if ~isfield(Parameters,'B0')
    warning('B0 field strength not specified, defaulting to 3 T');
    Parameters.B0 = 3.0;
end

% Check that the Matrix Size in Parameters is the right size
if isfield(Parameters,'MatrixSize')
    if Parameters.MatrixSize ~= size(arr_field)
        warning('Specified matrix size is not correct!');
        Parameters.MatrixSize = size(arr_field);
    end
else
    Parameters.MatrixSize = size(arr_field);
end

% Parameters
if ~isfield(Parameters,'Precond')
    Parameters.Precond = 30;
end

if ~isfield(Parameters,'Lambda')
    Parameters.Lambda = 1e-5;
end


%% PRE-PROCESSING

% Do a MIP through the magnitude image if multi-echo data was supplied
if size(arr_magn,4) > 1
    arr_magn = max(arr_magn,[],4);
end

% Check if there is a noise matrix supplied, and if not, use a "magnitude-based"
% data weighting
if isfield(Parameters,'Noise')
    arr_Wdata = 1./Parameters.Noise;
    arr_Wdata(isnan(arr_Wdata)) = 0;
    arr_Wdata(isinf(arr_Wdata)) = 0;
    arr_Wdata = arr_Wdata ./ prctile(arr_Wdata,90,"all");
else
    arr_Wdata = arr_magn ./ prctile(arr_magn,95,"all");
end
arr_Wdata(arr_Wdata > 1) = 1;

% Zero-pad the data, if necessary
Parameters.MatrixSize = max(Parameters.MatrixSize,[256,256,256]);
[arr_field, zpslices] = ZeroFilling(arr_field,Parameters.MatrixSize);
arr_magn = ZeroFilling(arr_magn,Parameters.MatrixSize);
arr_Wdata = ZeroFilling(arr_Wdata,Parameters.MatrixSize);
arr_mask = ZeroFilling(arr_mask,Parameters.MatrixSize);

% Zero-pad optional extras
if isfield(Parameters,'R2s')
    Parameters.R2s = ZeroFilling(Parameters.R2s,Parameters.MatrixSize);
end

%% GENERATE PRECONDITIONER AND WEIGHTS

% Basic Preconditioner
arr_P = ones(size(arr_mask));
arr_P(~arr_mask) = Parameters.Precond;

% Mask out high R2* valued voxels
if isfield(Parameters,'R2s')
    arr_P(Parameters.R2s > 100) = Parameters.Precond;
end

% Data weighting
arr_Wdata2 = arr_mask.*arr_Wdata.^2;

% Dipole kernel
arr_Dk = ifftshift(ak_Dk(Parameters));

% Initialize chi array
arr_chi = zeros(Parameters.MatrixSize);


%% MOVE THINGS TO GPU
if ~isfield(Parameters,'CPU') || Parameters.CPU ~= 1
    arr_P = gpuArray(single(arr_P));
    arr_field = gpuArray(single(arr_field));
    arr_magn = gpuArray(single(arr_magn));
    arr_mask = gpuArray(arr_mask);
    arr_Wdata2 = gpuArray(arr_Wdata2);
    arr_Dk = gpuArray(single(arr_Dk));
    arr_chi = gpuArray(single(arr_chi));
else
    disp('Not using the GPU');
end


%% GRADIENT WEIGHTS (on GPU)
arr_Mgrad = compute_grad_weights(arr_magn,arr_mask);


%% LOOP PARAMETERS

% Parameters for outer loop
max_iter_outer = 25;
num_iter_outer = 0;
reltol_update = 0.01;
delta = 1;

% Parameters for biconjugate gradient
max_iter_cg = 15;
tol_cg = 1e-6;


%% MAIN TFI ALGORITHM

while ((num_iter_outer < max_iter_outer) && (delta > reltol_update))

    % Define matrix for directional gradients
    grad_MGPy = zeros([Parameters.MatrixSize,3]);
    if ~isfield(Parameters,'CPU') || Parameters.CPU ~= 1
        grad_MGPy = gpuArray(grad_MGPy);
    end

    % Loop through directions and calculate gradient
    for dd = 1:3
        grad_MGPy(:,:,:,dd) = arr_Mgrad(:,:,:,dd) .* cdiff(arr_P.*arr_chi,dd);
    end

    % ISQR of MGPy
    arr_modM = 1./sqrt(sum(grad_MGPy.^2,4) + (1e-6.*arr_P));

    % Normalize MGPy
    grad_MGPy = abs(grad_MGPy).*repmat(arr_modM,[1,1,1,3]);

    % Current estimate of the field map
    DPy = real(ifftn(arr_Dk .* fftn(arr_P.*arr_chi)));

    % Right hand side of the equation
    b = arr_P.*real(ifftn(arr_Dk .* fftn(arr_Wdata2.*(arr_field - DPy))));

    % Subtract gradient component from b for each dimension
    for dd = 1:3
        bsub = Parameters.Lambda .* arr_P .* (cdiff(arr_Mgrad(:,:,:,dd) .* grad_MGPy(:,:,:,dd),dd));
        b = b - bsub;
    end

    % Set up CG operator
    mat_A = @(xx) (define_mat_sys(arr_Dk,arr_P,arr_Wdata2,arr_Mgrad,arr_modM,Parameters,xx));

    % Linear optimization using MATLAB's biconjugate gradient (stable) function
    [dy,flagcgs,residcgs,itercgs] = bicgstab(mat_A,b(:),tol_cg,max_iter_cg);
    fprintf('\tBICG stopped after %d iterations with residual %.4g (Flag %d)\n',itercgs,residcgs,flagcgs);

    % Reshape dy into an array
    dy = reshape(dy,size(arr_chi));

    % Update the susceptibility estimate
    arr_chi = arr_chi + dy;

    % Calculate delta
    delta = norm(dy(:)) ./ norm(arr_chi(:));

    % Update iteration counter
    num_iter_outer = num_iter_outer + 1;

    % Display output
    fprintf('\tIter: %d,\t delta: %.4g \n',num_iter_outer,delta);

end % while ((num_iter_outer < max_iter_outer) && (delta > reltol_update))


%% OUTPUT

% Mask the output and gather back from GPU
arr_susc = gather(arr_mask .* arr_chi);

% Undo zero-padding
arr_susc = arr_susc(zpslices{1},zpslices{2},zpslices{3});

end % function

%% DEFINE SYSTEM MATRIX
function mat_sys = define_mat_sys(arr_Dk,arr_P,arr_W,arr_M,arr_modM,Parameters,dy)
    % Define system matrix A for TFI conjugate gradient solving

    % Temporarily turn dy back into a matrix
    dy = reshape(dy,Parameters.MatrixSize);
        
    % Define LHS
    mat_sys = arr_P .* real(ifftn(arr_Dk .* fftn(arr_W .* real(ifftn (arr_Dk .* fftn(arr_P.*dy))))));

    % Some sort of gradient based regularization
    for dd = 1:3
        % Define spatial-gradient term
        MGPdyi = arr_M(:,:,:,dd) .* (cdiff(arr_P.*dy,dd));

        % Define regularization term
        sys_add = Parameters.Lambda .* arr_P .* (cdiff(arr_M(:,:,:,dd) .* arr_modM .* MGPdyi,dd));

        % Add to system matrix
        mat_sys = mat_sys + sys_add;
    end

    % Vectorize the output
    mat_sys = mat_sys(:);

    % % Tikhonov regularization
    % mat_sys = mat_sys + Parameters.Alpha.*arr_P.*dy;

end % function mat_sys = define_mat_sys(...)


%% COMPUTE GRADIENT WEIGHTS
function arr_edges = compute_grad_weights(arr_magn,arr_mask)
    % Calculate 3D gradients of magnitude

    % Pre-allocate
    arr_gw = zeros([size(arr_magn),3]);

    % Loop over directions
    for dd = 1:3

        % Compute absolute voxel-wise difference in the current direction
        arr_diff = abs(cdiff(arr_magn,dd));

        % Linearize and remove low-intensity voxels
        vec_diff = arr_diff(arr_magn(:) > 0.1);

        % Define percentiles (min, threshold, max)
        diff_min = prctile(vec_diff,1);
        diff_thr = prctile(vec_diff,95);
        diff_max = prctile(vec_diff,99);

        % Scale absolute value
        arr_diff = (arr_diff - diff_min) ./ (diff_max - diff_min);

        % Scale threshold value
        diff_thr = (diff_thr - diff_min) ./ (diff_max - diff_min);

        % Maximum-filter and scale
        arr_gw(:,:,:,dd) = max(diff_thr - arr_diff,0) ./ diff_thr;

    end

    % Set a lower threshold
    arr_gw(arr_gw < 0.1) = 0.1;

    % Apply the mask
    arr_edges = arr_gw .* repmat(arr_mask,1,1,1,3);

end % function arr_gw = compute_grad_weights(arr_magn)


%% CENTRAL DIFFERENCE
function arr_diff = cdiff(arr_input,dim)
    % Approximate gradient using central difference method

    % Rotate it so that the specified DIM is now in the first dimension
    arr_input = shiftdim(arr_input,dim-1);

    % Calculate central difference along the first dimension
    arr_diff = -0.5.*circshift(arr_input,1,1) + 0.5.*circshift(arr_input,-1,1);
    arr_diff(1,:,:) = -arr_input(1,:,:) + arr_input(2,:,:);
    arr_diff(end,:,:) = arr_input(end,:,:) - arr_input(end-1,:,:);

    % Rotate the matrix back to how it was before
    arr_diff = shiftdim(arr_diff,4-dim);

end % function arr_diff = cdiff(arr_input,dim)