from typing import Any
import numpy as np
import scipy.linalg as la

def OMP(signal, mD, sparsity, eps=1e-3):
    '''
    Orthogonal Matching Pursuit (OMP) algorithm

    Parameters
    ----------
    signal : numpy.ndarray
        The signal matrix
    mD : numpy.ndarray
        The dictionary matrix
    sparsity : int
        The sparsity of the sparse representation
    eps : float, optional
        The threshold of the residual, by default 1e-3

    Returns
    -------
    X : numpy.ndarray
        The sparse representation matrix
    '''
    _mX = []

    for i in range(signal.shape[1]):
        # OMP algorithm for the i-th signal
        # Initialize the sparse representation x = 0 & residual r = y
        _x = np.zeros(mD.shape[1])
        _r = signal[:, i].copy()

        for _ in range(sparsity+1):    # Iterate at most sparsity times
            _k = np.argmax(np.abs(mD.T @ _r))
            _x[_k] += mD[:, _k].T @ _r
            _r -= mD[:, _k] * (mD[:, _k].T @ _r)

            # Stop if |r|_{\infty} < eps
            if np.max(np.abs(_r)) < eps:
                break
        
        _mX.append(_x.reshape(-1, 1))   # Add to the sparse representation matrix

    return np.concatenate(_mX, axis=1)


class DICTLEARN:
    def __init__(self, alpha=0.5, beta=0.5, max_iter=1000, eps=1e-3):
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.eps = eps

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.DictLearn(*args, **kwds)


    def Phi(self, d, y, x):
        '''
        Phi(d) = ||y - d * x||_F^2 / 2
        '''
        return la.norm(y - d @ x)**2 / 2

    def GradPhi(self, d, y, x):
        '''
        GradPhi(d) = -2 * (y - d * x) * x.T
        '''
        return - (y - d @ x) @ x.T
    

    def ArmijoLineSearch(self, f, x0, dx):
        '''
        Armijo line search
        '''
        y0 = f(x0)

        t = 1

        for _ in range(self.max_iter):
            if f(x0 + t * dx) <= y0 - self.alpha * t * la.norm(dx)**2:
                break
            t *= self.beta

        return t, f(x0 + t * dx)


    def GD(self, d_, y_, x_):
        '''Gradient descent for optimization problem
        min_d Phi(d)
        '''
        f = lambda d: self.Phi(d, y_, x_)

        for _ in range(100):
            d_tmp = d_
            phi_tmp = f(d_tmp)
            grad_d_ = self.GradPhi(d_, y_, x_)

            t, phi = self.ArmijoLineSearch(f, d_, -grad_d_)
            d_ = d_ - t * grad_d_

            if la.norm(d_tmp - d_) < self.eps or np.abs((phi - phi_tmp) / phi_tmp) < self.eps:
                break
        
        return d_


    def DictLearn(self, mD, mX, patches):
        for i in range(mD.shape[1]):
            indices = np.where(mX[i, :] != 0)[0]

            if indices.size == 0:
                continue

            y_ = patches[:, indices]
            x_ = mX[i, indices].reshape(1, -1)
            d_ = mD[:, i].reshape(-1, 1)

            d_ = self.GD(d_, y_, x_).reshape(-1)
            mD[:, i] = d_ / la.norm(d_)
        
        return mD


def KSVD(patches, dictionary, sparsity, _iteration=10, eps=1e-6):
    '''
    K-SVD algorithm

    Parameters
    ----------
    patches : numpy.ndarray
        The patches
    sparsity : int
        The sparsity
    dictionary : numpy.ndarray
        The dictionary
    max_iter : int
        The maximum number of iterations
    eps : float
        The threshold

    Returns
    -------
    mD : numpy.ndarray
        The dictionary
    mX : numpy.ndarray
        The sparse representation
    '''
    DictLearn = DICTLEARN(eps=eps)

    mD = dictionary
    mX = OMP(patches, mD, sparsity, eps)

    for _ in range(_iteration):
        mD = DictLearn(mD, mX, patches)
        mX = OMP(mD, patches, sparsity, eps)
    
    return mD @ mX, mD, mX
