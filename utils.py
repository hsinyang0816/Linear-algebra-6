from PIL import Image
import numpy as np
from utils import *
import matplotlib.pyplot as plt


def load_image(img_path):
    """Load image into a 3D numpy array
    Arg:
        img_path: string, file path of the image file.
    Return:
        imArr: numpy array with shape (height, width, 3).
    """
    #return img_path
    im = Image.open(img_path).convert('RGB') #as im:
    #print(im.size)
    #s()
    imArr = np.fromstring(im.tobytes(), dtype=np.uint8)
    imArr = imArr.reshape((im.size[1], im.size[0], 3))
    return imArr
    

def save_image(imArr, fpath='output.png'):
    """Save numpy array as a png file
    Arg:
        imArr: 2d or 3d numpy array, *** it must be np.uint8 and range from [0, 255]. ***
        fpath: string, the path to save imgArr.
    """
    im = Image.fromarray(imArr)
    im.save(fpath)
    
def plot_curve(k, err, fpath='curve.png', show=False):
    """Save the relation curve of k and approx. error to fpath
    Arg:
        k: a list of k, in this homework, it should be [1, 5, 50, 150, 400, 1050, 1289]
        err: a list of aprroximation error corresponding to k = 1, 5, 50, 150, 400, 1050, 1289
        fpath: string, the path to save curve
        show: boolean, if True: display the plot else save the plot to fpath
    """
    plt.gcf().clear()
    plt.plot(k, err, marker='.')
    plt.title('SVD compression')
    plt.xlabel('k')
    plt.ylabel('Approx. error')
    if show:
        plt.show()
    else:
        plt.savefig(fpath, dpi=300)

def approx_error(imArr, imArr_compressed):
    """Calculate RMSE approximation error 
    Arg:
        Two numpy arrays
    Return:
        A float number, approximation error
    """
    v = imArr.ravel().astype(float)
    u = imArr_compressed.ravel().astype(float)
    return np.linalg.norm(v - u) / np.sqrt(len(v)) / 255
