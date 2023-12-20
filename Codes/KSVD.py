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

    def __call__(self, patches, mD, mX):
        return self.DictLearn(patches, mD, mX)


    def DictLearn(self, patches, mD, mX):
        for i in range(mD.shape[1]):
            _indices = np.where(mX[i, :] != 0)[0]

            if _indices.size == 0:
                continue

            _mY_ = patches[:, _indices]
            _d = mD[:, i].reshape(-1, 1)
            _mX_ = mX[i, _indices].reshape(1, -1)

            _d = self.GD(_d, _mY_, _mX_).reshape(-1)
            mD[:, i] = _d / la.norm(_d)
        
        return mD
    
    
    def GD(self, d, mY_, mX_):
        '''Gradient descent for optimization problem
        min_d Phi(d)
        '''
        f = lambda d: self.Phi(d, mY_, mX_)

        for _ in range(self.max_iter):
            _d_tmp, _phi_tmp = d, f(d)
            _gd = self.GradPhi(d, mY_, mX_)

            d, phi = self.ArmijoLineSearch(f, d, -_gd)

            if la.norm(_d_tmp - d) < self.eps or np.abs((phi - _phi_tmp) / _phi_tmp) < self.eps:
                break
        
        return d
    

    def Phi(self, d, mY_, mX_):
        '''
        Phi(d) = ||Y(d) - d * X(d)||_F^2 / 2
        '''
        return la.norm(mY_ - d @ mX_)**2 / 2

    def GradPhi(self, d, mY_, mX_):
        '''
        GradPhi(d) = - (Y(d) - d * X(d)) * x.T
        '''
        return - (mY_ - d @ mX_) @ mX_.T
    
    def ArmijoLineSearch(self, f, x0, dx):
        y0 = f(x0)

        t = 1

        for _ in range(self.max_iter):
            if f(x0 + t * dx) <= y0 - self.alpha * t * la.norm(dx)**2:
                break
            t *= self.beta

        _x = x0 + t * dx
        return _x, f(_x)


def KSVD(patches, dictionary, sparsity, _iteration=10, eps=1e-6):
    '''
    K-SVD algorithm

    Parameters
    ----------
    
    Returns
    -------
    
    '''
    DictLearn = DICTLEARN(eps=eps)

    mD = dictionary

    for _ in range(_iteration):
        if _ > 0:
            mD = DictLearn(patches, mD, mX)
        mX = OMP(patches, mD, sparsity, eps)
    
    return mD, mX
