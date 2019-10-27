import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from cpu_dist import dist
from sklearn.metrics.pairwise import euclidean_distances
import time
#from gpu_dist import dist as gdist


def e_dist(a, b, metric='euclidean'):
    """Distance calculation for 1D, 2D and 3D points using einsum
    : a, b   - list, tuple, array in 1,2 or 3D form
    : metric - euclidean ('e','eu'...), sqeuclidean ('s','sq'...), 
    :-----------------------------------------------------------------------
    """
    a = np.asarray(a)
    b = np.atleast_2d(b)
    a_dim = a.ndim
    b_dim = b.ndim
    if a_dim == 1:
        a = a.reshape(1, 1, a.shape[0])
    if a_dim >= 2:
        a = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    if b_dim > 2:
        b = b.reshape(np.prod(b.shape[:-1]), b.shape[-1])
    diff = a - b
    dist_arr = np.einsum('ijk,ijk->ij', diff, diff)
    if metric[:1] == 'e':
        dist_arr = np.sqrt(dist_arr)
    dist_arr = np.squeeze(dist_arr)
    return dist_arr

a = np.random.rand(10000,3)
b = np.random.rand(10000,3)




#print(np.allclose(pairwise_distances(a,b, 'sqeuclidean'), dist(a,b), atol=1e-5))
#print(a, b)

#a = np.asarray([[1 , 2 , 3],[1,2,3]])
#b = np.asarray([[3 , 4 , 5],[3,4,5]])


start_dist = time.time()
print(e_dist(a,a))
end_dist = time.time()


z = a - a
print(np.sqrt(np.einsum('ij,ij->i', z, z)))


start = time.time()
print(euclidean_distances(a, a))
end = time.time()
#print(dist(c,d))
#print(np.allclose(pairwise_distances(a,b, 'sqeuclidean'), gdist(a,b), atol=1e-5))

#a = np.random.rand(800,2048).astype(np.float32)
#b = np.random.rand(800,2048).astype(np.float32)

ss = time.time()
print(dist(a,a))
ee = time.time()




print(end_dist-start_dist, end-start, ee-ss)

#print(pairwise_distances(a,b, 'sqeuclidean'))
#print(np.allclose(pairwise_distances(a,b, 'sqeuclidean'), dist(a,b), atol=1e-5))


#X = [[1.5214, 1.5214]]
# distance between rows of X
#print(euclidean_distances(X, X))


# get distance to origin
#print(euclidean_distances(X, [[1.9350, 1.9350]]))
