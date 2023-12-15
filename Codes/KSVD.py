import numpy as np
import scipy.linalg as la

def OMP(dictionary, sample, sparsity, eps=1e-3):
    '''
    Orthogonal Matching Pursuit

    Parameters
    ----------
    dictionary : numpy.ndarray
        The dictionary
    sample : numpy.ndarray
        The sample
    sparsity : int
        The sparsity
    eps : float
        The threshold

    Returns
    -------
    x : numpy.ndarray
        The sparse representation
    '''
    # Initialization
    mX_ = np.zeros((dictionary.shape[1], sample.shape[1]))

    for i in range(sample.shape[1]):
        sample_ = sample[:, i]     # The sample
        coding = mX_[:, i]        # The sparse representation

        # Initialize the residual
        r = sample_
        iteration = 0
        
        while iteration <= sparsity:
            j = np.argmax(np.abs(dictionary.T @ r))
            coding[j] += dictionary[:, j].T @ r
            r = r - dictionary[:, j] * (dictionary[:, j].T @ r)

            if la.norm(r) < eps:    # If the residual is small enough
                break

            iteration += 1
        
        mX_[:, i] = coding    # Update the sparse representation
    
    return mX_


def ArmijoLineSearch(f, x0, dx, alpha=0.3, beta=0.5, max_iter=1000):
    '''
    Armijo line search
    '''
    y0 = f(x0)

    t = 1

    for _ in range(max_iter):
        if f(x0 + t * dx) <= y0 - alpha * t * la.norm(dx)**2:
            break
        t *= beta

    return t, f(x0 + t * dx)


def GD(d_, y_, x_, Phi, GradPhi):
    '''Gradient descent for optimization problem
    min_d Phi(d)
    '''
    f = lambda d: Phi(d, y_, x_)

    for _ in range(100):
        d_tmp = d_
        phi_tmp = f(d_tmp)
        grad_d_ = GradPhi(d_, y_, x_)

        t, phi = ArmijoLineSearch(f, d_, -grad_d_)
        d_ = d_ - t * grad_d_

        if la.norm(d_tmp - d_) < 1e-6 or np.abs((phi - phi_tmp) / phi_tmp) < 1e-6:
            break
    
    return d_


def DictLearn(mD, mX, patches, Phi, GradPhi):
    for i in range(mD.shape[1]):
        indices = np.where(mX[i, :] != 0)[0]

        if indices.size == 0:
            continue

        y_ = patches[:, indices]
        x_ = mX[i, indices].reshape(1, -1)
        d_ = mD[:, i].reshape(-1, 1)

        d_ = GD(d_, y_, x_, Phi, GradPhi).reshape(-1)
        mD[:, i] = d_ / la.norm(d_)
    
    return mD
