import json
import numpy as np

def convert(line: str):
    i = 0
    imUrl = ""
    cnt = 0
    while i < len(line) and cnt < 3:
        if line[i] == "'":
            if i + 5 < len(line) and line[i + 1: i + 5] == "asin":
                j = i + 8
                k = j + 1
                while line[k] != "'":
                    k += 1
                asin = line[j + 1:k]
                i = k + 1
                cnt += 1
            elif i + 6 < len(line) and line[i + 1:i + 6] == "imUrl":
                j = i + 9
                k = j + 1
                while line[k] != "'":
                    k += 1
                imUrl = line[j + 1:k]
                i = k + 1
                cnt += 1
            elif i + 11 < len(line) and line[i + 1:i + 11] == "categories":
                j = i + 14
                k = j + 1
                cnt = 1
                while 1:
                    if line[k] == "]":
                        cnt -= 1
                    elif line[k] == "[":
                        cnt += 1
                    if cnt == 0:
                        break
                    k += 1
                categories = line[j:k + 1]
                i = k + 1
                cnt += 1
            else:
                i = i + 1
        else:
            i = i + 1
    dic = {
        "asin": asin,
        "imUrl": imUrl,
        "categories": categories
    }
    return json.dumps(dic)

items = {}
reviews = {}
with open("reviews_Office_Products_5.json", "r") as f:
    data = f.readlines()
for line in data:
    line_data = json.loads(line)
    reviews[(line_data['reviewerID'], line_data['asin'])] = line_data['overall']
np.save("reviews.npy", reviews)
reviews = np.load("reviews.npy", allow_pickle=True).item()

with open("meta_Office_Products.json", "r") as f:
    data = f.readlines()
cnt = 0
error = 0
for line in data:
    cnt += 1
    try:
        line_c = convert(line)
        line_data = json.loads(line_c)
        items[line_data['asin']] = {"imUrl": line_data['imUrl'], 'categories': line_data['categories']}
    except Exception as e:
        error += 1
np.save("items.npy", items)
items = np.load("items.npy", allow_pickle=True).item()
print("total error =", str(error))
