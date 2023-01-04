import numpy as np
import random

reviews = np.load("reviews.npy", allow_pickle=True).item()
items_meta = np.load("items.npy", allow_pickle=True).item()
user_cnt = 0
item_cnt = 0
train_cnt = 0
test_cnt = 0
users = {}
items = {}
categories_idx = {}
categories_cnt = 0

train_f = open("amazon/train.txt", 'w')
test_f = open("amazon/test.txt", "w")

for key in items_meta.keys():
    try:
        categories = list(eval(items_meta[key]['categories']))[0]
        for category in categories:
            if category not in categories_idx:
                categories_idx[category] = categories_cnt
                categories_cnt += 1
    except:
        print(items_meta[key]['categories'])
        exit(1)

print("categories_cnt =", categories_cnt)

for key in reviews.keys():
    user = key[0]
    item = key[1]
    if user not in users:
        users[user] = user_cnt
        user_cnt += 1
    if item not in items:
        items[item] = item_cnt
        item_cnt += 1
    user = users[user]
    item = items[item]
    res = str(user) + ' ' + str(item) + ' 1\n'
    if random.random() > 0.1:
        train_cnt += 1
        train_f.write(res)
    else:
        test_cnt += 1
        test_f.write(res)
print("user cnt =", user_cnt)
print("item cnt =", item_cnt)

user_feature = np.zeros([user_cnt, categories_cnt])

for key in reviews.keys():
    user = users[key[0]]
    item = key[1]
    categories = list(eval(items_meta[item]['categories']))[0]
    for category in categories:
        user_feature[user][categories_idx[category]] += 1

np.save("feature.npy", user_feature)

train_f.close()
test_f.close()

print("train_cnt =", train_cnt)
print("test_cnt =", test_cnt)