import numpy as np


def svd_compress(imArr, K=37):
    """Compress image array using SVD decomposition.
    Arg:
        imArr: numpy array with shape (height, width, 3).
    Return:
        Compressed imArr: numpy array.
    """
    imArr_compressed = np.zeros(imArr.shape)
    # For each channel
    for ch in range(3):
        # --------------------
        # TODO: 
        #     Compress the image array using SVD decomposition
        # hint:
        #     1. np.linalg.svd
        #     2. np.diag
        #     3. np.dot
        #
        # Your code here
        # ?? = imArr[:,:,ch]
        # imArr_compressed = ??
        # 
        # --------------------
        # clip image to 0 ~ 255, DO NOT MODIFY THIS PART
        U,sigma,VT = np.linalg.svd(imArr[:,:,ch])
        
        # print("U :   ")
        # print(U[:,0:50])
        new_U = np.zeros(U.shape)
        new_U[:,0:K]=U[:,0:K]
        # print("new_U :   ")
        # print(new_U[:,0:50])
        new_sigma = np.zeros(sigma.shape)
        new_sigma[0:K]=sigma[0:K]
        new_sigma = np.diag(new_sigma)
        if(U.shape[0] > VT.shape[0]):
            new_sigma = np.vstack((new_sigma, np.zeros((U.shape[0] - VT.shape[0], VT.shape[0]))))
        else:
            new_sigma = np.hstack((new_sigma, np.zeros((U.shape[0], VT.shape[0] - U.shape[0]))))
        new_VT = np.zeros(VT.shape)
        new_VT[0:K,:]=VT[0:K,:]

        imArr_compressed[:,:,ch] = np.dot(new_U, np.dot(new_sigma, new_VT))
        # print(imArr_compressed.shape)
        imArr_compressed[:, :, ch] = np.clip(imArr_compressed[:, :, ch],0,255)
    #imArr_compressed[:, :, 0] = imArr_compressed[:, :, 1]
    #print(np.count_nonzero(imArr_compressed))
    return imArr_compressed.astype(np.uint8)