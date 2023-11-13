curdir = pwd; cd /home/jfnielse/github/mirt/; setup; cd(curdir);
%addpath ~/github/tianrluo/AutoDiffPulses/
%addpath /home/jfnielse/github/jfnielsen/AutoDiffPulses/   % my fork, for adding new costs
%addpath ~/Downloads/AutoDiffPulses-master
setup_AutoDiffPulses;            % AutoDiff setup
%addpath /home/jfnielse/github/tianrluo/       % +mrphy and +attr

gpuID = 2;  % View current GPU usage with `nvtop`  

smax = 20e3; 

%input('Set dt = 20e-3 and smax = 20 in penalties.py and press any key to continue');

% Initialize gradients and RF pulse 
if false
    system('python sp3d.py 0.8');  % a hand-coded 3D spiral that works fairly well
    load g.mat g dt
else
    dur = 1; % ms
    %dt = 20e-3;
    dt=4e-3;
    g = spins(5, 0.5, dur, 12*pi, 2*pi, 6, 0.8, dt);
    g = g/2;  % seems to work better
end

nt = size(g,2);

%rf = single(0.00*ones(1,nt)); % initialize with zeros
rf=rand(1,nt);
pIni = mrphy.Pulse('rf', rf, 'gr', g, 'dt', dt*1e-3, ...
    'gmax', 5.0, 'smax', smax, 'rfmax', 0.25, 'desc', '3d spiral');

%% Define iv and ov regions
fov = 24*[1 1 1];  % cm
nx = 60; ny = 60; nz = 60;
%nx=32;ny=32;nz=32;
%[x, y, z] = ndgrid( linspace(-1, 1, nx), linspace(-1, 1, ny), linspace(-1, 1, nz));
[x, y, z] = ndgrid( linspace(-4/3, 4/3, nx), linspace(-4/3, 4/3, ny), linspace(-4/3, 4/3, nz));
r = sqrt(x.^2 + y.^2 + z.^2);
objectmask = r < 1.0;  % object support
dy = 0;  % cm
c = [0 dy/(fov(2)/2) 0];    % iv center (units of fov)
rc = sqrt((x-c(1)).^2 + (y-c(2)).^2 + (z-c(3)).^2); % distance from center of iv
iv = rc < 0.3;    % iv diameter
se = strel('sphere',2);
tmp = imdilate(iv, se);
ov = objectmask & ~tmp;        % outer volume
mask = (iv | ov);

%% Initialize cube object
ofst = [0 0 0]; % offset, I think...
cube = mrphy.SpinCube(fov, [nx ny nz], ofst, 'mask', mask);
cubesim = mrphy.SpinCube(fov, [nx ny nz], ofst, 'mask', objectmask);  % for simulation
%%

%% Target
% % Note that ml2xy error metric has been repurposed slightly
% flipsat = 60;
% d = zeros(nx, ny, nz, 3);
% d(:,:,:,1) = sind(flipsat) * iv;
% d(:,:,:,3) = cosd(flipsat) * iv + 1.0 * ov;
% target.d = d;
% target.weight = 0.2*ov + 1.0*iv;
%%

%% SS Target
flipsat = 60;%saturation angle
alpha=15;%excitation angle
t1 = 1300;  % ms
tr = 55;    % ms
E1=exp(-tr/t1);
Mz_iv=(1-E1)/(1-cosd(flipsat)*cosd(alpha)*E1);%steady state Mz in iv
Mz_ov=(1-E1)/(1-cosd(alpha)*E1);%SS Mz in ov 
d = zeros(nx, ny, nz, 3);
d(:,:,:,3) = Mz_iv * iv + Mz_ov* ov;
d(:,:,:,1) = cosd(flipsat) *sind(alpha)*Mz_iv*iv+sind(alpha)*Mz_ov*ov ;
target.d = d;
target.weight = 0.2*ov + 1.0*iv;

%% Design IVsat pulse 
pADsat_ss = adpulses.opt.arctanAD_ss(target, cube, pIni, ... %'rasteroptim', ...
    'niter', 30, 'niter_rf', 2, 'niter_gr', 1, ...
    'err_meth', 'l2z', 'doClean', false, 'gpuID', gpuID);

save pADsat_bfgs pADsat_ss

% mz = plot_res(pIni, pADsat, cubesim, target, 'z', false);
% 
% % Calculate steady-state mz due to sat pulse Ignore effect of excitation
% % pulse on steady-state mz for now.
% fa = acos(mz)/pi*180;   % flip angle corresponding to mz
% t1 = 1300;  % ms
% tr = 55;    % ms
% mxy_ss = spgr(tr, t1, fa);  % steady-state transverse magnetization
% mz_ss = mxy_ss./sind(fa);   % steady-state longitudinal magnetization
% figure; im(mz_ss); title('mz_ss');

return
