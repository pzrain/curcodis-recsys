import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
features = np.array(np.load("feature.npy", allow_pickle=True))
features = features[:,1:]
user_cnt = features.shape[0]
edge_cnt = 0

trust_f = open("amazon/trust.txt", "w")

s = cosine_similarity(features)
threshold = 0.905
for i in range(user_cnt):
    for j in range(i + 1, user_cnt):
        if s[i][j] > threshold:
            # print(s[i][j])
            res = str(i) + ' ' + str(j) + '\n'
            trust_f.write(res)
            edge_cnt += 1

trust_f.close()
print("edge_cnt =", edge_cnt)
