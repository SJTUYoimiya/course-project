import numpy as np
import matplotlib.pyplot as plt

'''
This file is used to transform the image into patches and vice versa.
'''

class IMG2PATCH:
    def __init__(self):
        pass
    

    def __call__(self, *args, **kwargs):
        if kwargs.get('inv'):
            return self.Patch2Img(*args)
        else:
            return self.Img2Patch(*args)
    

    def Img2Patch(self, img, d, s):
        '''
        Divide the image into patches

        Parameters
        ----------
        img : ndarray
            The image to be divided.
        d : int
            The _size of the patch.
        s : int
            Minimum number of overlapping pixels between patches.

        Returns
        -------
        patches : ndarray
            The patches of the image.
        locs : ndarray
            The locations of the patches.
        dcs : ndarray
            The DC components of the patches.
        '''
        h, l = img.shape    # Get the _size of the image (height, length)

        # Calculate the coordinates of the upper left corner of patch
        _nh = (h - s) // (d - s) + 1
        _nl = (l - s) // (d - s) + 1
        loc_y = np.linspace(0, h - d, _nh).astype(int)
        loc_x = np.linspace(0, l - d, _nl).astype(int)

        _locs = np.meshgrid(loc_x, loc_y)
        _loc_x = _locs[0].flatten()
        _loc_y = _locs[1].flatten()

        # Initialization
        _patches = []
        _dcs = []

        for i in range(len(_loc_x)):
            # Get the patch
            _x = _loc_x[i]
            _y = _loc_y[i]
            _patch = img[_y: _y+d, _x: _x+d].reshape(-1, 1)

            # extract the DC component
            _dc = np.mean(_patch)
            _dcs.append(_dc)
            _patch = _patch - _dc
            _patches.append(_patch)

        _patches = np.concatenate(_patches, axis=1)
        return _patches, [loc_x, loc_y], _dcs
    
    
    def Patch2Img(self, patches, locs, dcs):
        '''
        Combining patches into an image

        Parameters
        ----------
        patches : ndarray
            The patches of the image.
        locs : ndarray
            The locations of the patches.
        dcs : ndarray
            The DC components of the patches.

        Returns
        -------
        img : ndarray
            The image.
        '''
        # Reconstruct the image
        _d = int(patches.shape[0]**0.5)

        _loc_x, _loc_y = locs
        _nl = len(_loc_x)
        _nh = len(_loc_y)
        _l = _loc_x[-1] + _d
        _h = _loc_y[-1] + _d

        _img = np.zeros((_h, _l))
        # Put the patch back in its place
        for idx in range(patches.shape[1]):
            _patch = patches[:, idx].reshape(_d, -1)    # Transform the column vector into a matrix
            _patch = _patch + dcs[idx]                  # Add the DC component

            # Locate the patch
            _x = _loc_x[idx % _nl]
            _y = _loc_y[idx // _nl]
            _img[_y:_y+_d, _x:_x+_d] += _patch


        # Average the overlapping area
        for j in range(1, _nl):
            _img[:, _loc_x[j]:_loc_x[j-1]+_d] = _img[:, _loc_x[j]:_loc_x[j-1]+_d] / 2

        for i in range(1, _nh):
            _img[_loc_y[i]:_loc_y[i-1]+_d, :] = _img[_loc_y[i]:_loc_y[i-1]+_d, :] / 2

        return _img


Img2patch = IMG2PATCH()


def Show(*args, cmap='gray'):
    args = list(args)   
    for i, arg in enumerate(args):
        _min = np.array([arg.min(), 0]).min()
        _max = np.array([arg.max(), 1]).max()
        arg = (arg - _min) / (_max - _min)
        args[i] = arg

    if cmap == 'gray' :
        plt.imshow(*args, cmap=cmap)
        plt.axis('off')
        plt.show()
    else:
        RGBs = ['r', 'g', 'b']
        _size = args[0].shape
        _img = np.zeros((*_size, 3))

        for i, _channel in enumerate(list(cmap)):
            _idx = RGBs.index(_channel)
            _img[:, :, _idx] = args[i]
        
        plt.imshow(_img)
        plt.axis('off')
        plt.show()
        return _img


if __name__ == '__main__':
    img = plt.imread('./Images/McM images/McM13.tif')
    Show(img[:, :, 2], cmap='b')
    # print(img[:, :, 0].shape)
