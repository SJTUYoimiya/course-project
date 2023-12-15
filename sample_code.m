close; clear; clc

addpath ./Pictures/
%%
filename = 'lena.tif';

u = double(imread(filename));
u = u(1:128, 1:128, :);

u0 = u;

u = u + 1e-2* randn(size(u));

n = size(u, 1:2);
p = [8, 8];

%%%% dictionary
nA = 64; % number of Atoms

%%%% image to patches
np = prod(p); % number of pixels

%%%% number of patches
gap = 4; % control the overlapping
reS = n(1)/gap - 5;
I = round(linspace(1, n(1)-p(1)+1, reS));
J = round(linspace(1, n(2)-p(2)+1, reS));

nP = numel(I)* numel(J); % total number of patches
iP = zeros(nP, 5);


%%%% extracting the patches
X = zeros(np, nP);
c = 1;
for j = 1:numel(J)
    range_j = J(j): J(j)+p(2) -1;
    for i=1:numel(I)
        range_i = I(i): I(i)+p(1)-1;
        
        iP(c, :) = [c, range_i(1), range_i(end), range_j(1), range_j(end)];

        patch = u(range_i, range_j);
        patch_ = patch(:) - mean(patch(:)); % remove the DC part
        
        X(:, c) = patch_; % - mean(patch(:));

        c = c + 1;
    end
end

% size(A) = [K, N];
%% initial dictionary and coefficient
% you can design other starting points

D0 = eye(np, nA);
A0 = repmat(D0(:, 1:np)\ X, nA/np, 1);
%%
% the code below is for solving the following problem
% min_{D, A} 0.5||X - DA||_F^2 + \lambda ||A||_1
% subject to ||D_i||_2 \leq 1, i=1,...,nA
% the constraint is that each atom in thd dictionary has unit length
% where l1-norm of A is used, no smoothing is considered.

lambda = 15; %% regularization parameter

tol = 1e-4; %% tolerance for algorithm to stop
maxits = 1e4; %% maximum number of interation

ek = zeros(maxits, 1); %% record residual error

D = D0;
A = A0;

its = 1;
while its<=maxits-1
    D_old = D;
    A_old = A;

    %%% gradient descent on D
    L_D = norm(A*A') + .1;
    grad_D = (-X + D*A)* (A');

    D = D - (1.1/L_D)* grad_D;
    
    %%% enforcing unit length on to the atoms
    for j=1:nA
        D(:, j) = D(:, j) / norm(D(:,j));
    end

    %%% proximal gradient descent on A
    L_A = norm(D*D') + .1;
    gamma_A = 1.1/L_A;

    grad_A = (D')* (-X + D*A);
    w = A - gamma_A* grad_A;

    %%% this is for dealing the l1-norm, called soft-thresholding shrinkage
    A = sign(w) .* max(abs(w)-lambda*gamma_A, 0);

    ek(its) = norm(D(:)-D_old(:));

    if ek(i) < tol || ek(i) > 1e10; break; end

    its = its + 1;
    if mod(its,20)==0; fprintf('%06d...\n', its); end
end

%% output the dictionary learned
D0_ = D0;
I1 = 8;
J1 = nA/I1;

p_ = p + 1;
imD0 = -0* ones([I1, J1] .* p_);
imD = -0* ones([I1, J1] .* p_);

c = 0;
for i=1: I1
    range_i = (i-1)*p_(1) + 1 : i*p_(1)-1;
    for j = 1: J1
        c = c + 1;

        range_j = (j-1)*p_(2) + 1 : j*p_(2)-1;

        patch = D0_(:, c);
        imD0(range_i, range_j) = reshape(patch, p) - min(patch(:));

        patch = D(:, c);
        imD(range_i, range_j) = reshape(patch, p) - min(patch(:));
    end
end

