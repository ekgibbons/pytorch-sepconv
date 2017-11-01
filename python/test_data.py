import numpy as np
from matplotlib import pyplot as plt

import GenerateHeartData

pathRead = "/v/raid1a/egibbons/data/deep-slice"

for ii in range(10):
    data = np.load("%s/training_hearts_%02i.npy" % (pathRead, ii))
    print(ii)
    X, y = GenerateHeartData.DataGenerator(data)

    montage = np.concatenate((X[1000,:,:,0],y[1000,:,:,0],X[1000,:,:,1]),axis=1)
    
    plt.figure()
    plt.imshow(montage)
    plt.show()
