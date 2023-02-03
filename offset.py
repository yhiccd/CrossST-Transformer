import numpy as np
util = np.load('skeleton.npz')
skeleton = util['skeleton']
parent = util['parent']
offset = []
for i in range(skeleton.shape[0]):
    if i != 0:
        temp = skeleton[i] - skeleton[parent[i][0]]
    else:
        temp = skeleton[i]
    
    offset.append(temp)

np.savez('skeleton.npz', skeleton = skeleton, parent = parent, offset = offset)