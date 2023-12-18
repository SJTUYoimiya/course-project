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
            return self.patch2img(*args)
        else:
            return self.img2patch(*args)
    

    def img2patch(self, img, size, overlapping_rate):
        '''
        Transform the image into patches

        Parameters
        ----------
        img : numpy.ndarray
            The input image
        size : int
            The size of the patch -- P = d**2
        overlapping_rate : float
            The rate of the overlapping area -- overlapping_rate = d_{overlap} / d
            -> h = d * (1 - repeat_rate)

        Returns
        -------
        patches : numpy.ndarray (shape = (P, N))
            The patches of the image
        '''
        img_size = img.shape[0] # Number of pixels on each side

        # Calculate the upper left endpoint of patches
        d_ = int(size**0.5)
        h_ = d_ * (1 - overlapping_rate)
        endpoints_ = np.arange(0, img_size-d_+h_, h_).astype(int)

        # Initialization
        patches_ = []
        DC_component_ = []

        for i in range(len(endpoints_)):
            pi_ = endpoints_[i]
            for j in range(len(endpoints_)):
                pj_ = endpoints_[j]

                patch_ = img[pi_:pi_+d_, pj_:pj_+d_]    # Get the patch[i, j] with size = d * d
                patch_ = patch_.reshape(-1, 1)          # concatenate the patch[i, j] into a column vector

                # Subtract & record the DC component from patch[i, j]
                DC_component_.append(np.mean(patch_))
                patch_ = patch_ - np.mean(patch_)
                patches_.append(patch_)

        patches_ = np.concatenate(patches_, axis=1)
        return patches_, DC_component_
    
    
    def patch2img(self, patches, overlapping_rate, DC_component):
        '''
        Combining patches into an image

        Parameters
        ----------
        patches : numpy.ndarray (shape = (P, N))
            The patches of the image
        overlapping_rate : float
            The rate of the overlapping area
        DC_component : numpy.ndarray
            The DC component of patches

        Returns
        -------
        img_ : numpy.ndarray (shape = (img_size_, img_size_))
            The image
        '''
        # Calculate the size of the image
        P, N = patches.shape
        d_ = int(P**0.5)
        n_ = int(N**0.5)
        img_size_ = int(d_ * (n_ - (n_ - 1) * overlapping_rate))

        img_ = np.zeros((img_size_, img_size_)) # Initialize the image

        # Put the patch back in its place
        for idx in range(N):
            patch_ = patches[:, idx].reshape(d_, d_)    # Transform the column vector into a matrix
            # Locate the patch
            i_ = int((idx // n_) * d_ * (1 - overlapping_rate))
            j_ = int((idx % n_) * d_ * (1 - overlapping_rate))
            
            patch_ = patch_ + DC_component[idx]         # Add the DC component

            img_[i_:i_+d_, j_:j_+d_] = img_[i_:i_+d_, j_:j_+d_] + patch_    # Add the patch to the image

        # Average the overlapping area
        for j in range(1, n_):
            p_ = int(j * d_ * (1 - overlapping_rate))
            img_[:, p_:p_+int(d_ * overlapping_rate)] = img_[:, p_:p_+int(d_ * overlapping_rate)] / 2

        for i in range(1, n_):
            p_ = int(i * d_ * (1 - overlapping_rate))
            img_[p_:p_+int(d_ * overlapping_rate), :] = img_[p_:p_+int(d_ * overlapping_rate), :] / 2

        return img_


Img2patch = IMG2PATCH()


def Showing(*args, cmap='gray'):
    RGBs = ['Red Channel', 'Green Channel', 'Blue Channel']

    if len(args) == 3 or cmap in ['rgb', 'RGB']:
        size = args[0].shape
        img_ = np.zeros((*size, 3))
        for color in range(3):
            img_[:, :, color] = args[color]
        
        img_ = img_.astype(int)
        plt.imshow(img_)
    elif cmap == 'gray' :
        plt.imshow(*args, cmap=cmap)
    elif cmap in RGBs:
        size = args[0].shape
        img_ = np.zeros((*size, 3))

        idx = RGBs.index(cmap)
        img_[:, :, idx] = np.array(args)

        img_ = img_.astype(int)
        plt.imshow(img_)

    plt.axis('off')
    plt.show()
