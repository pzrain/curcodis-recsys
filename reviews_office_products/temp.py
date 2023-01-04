import json
def convert(line: str):
    i = 0
    imUrl = ""
    while i < len(line):
        if line[i] == "'":
            if i + 5 < len(line) and line[i + 1: i + 5] == "asin":
                j = i + 8
                k = j + 1
                while line[k] != "'":
                    k += 1
                asin = line[j + 1:k]
                i = k + 1
            elif i + 6 < len(line) and line[i + 1:i + 6] == "imUrl":
                j = i + 9
                k = j + 1
                while line[k] != "'":
                    k += 1
                imUrl = line[j + 1:k]
                i = k + 1
            elif i + 11 < len(line) and line[i + 1:i + 11] == "categories":
                j = i + 14
                k = j + 1
                cnt = 0
                while 1:
                    if line[k] == "]":
                        cnt += 1
                    if cnt == 2:
                        break
                    k += 1
                categories = line[j:k + 1]
                i = k + 1
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
print(convert("{'asin': 'B00006IEBI', 'categories': [['Office Products', 'Office & School Supplies', 'Writing & Correction Supplies', 'Pens & Refills', 'Rollerball Pens']]}"))