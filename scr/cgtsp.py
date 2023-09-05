import random

d = {0:[[0]], 1:[[1]], 2:[[2,3], [4,5,6]], 3:[[7,8], [9,10,11], [12]]}
num_cluster = 4
lst = []
for i in range(100):
    cluster_start = [0]
    cluster_random = list(range(1, num_cluster))
    random.shuffle(cluster_random)
    lst.append(cluster_start + cluster_random)

