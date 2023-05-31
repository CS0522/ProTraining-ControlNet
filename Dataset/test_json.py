import json

f = open("./prompt.json", "r")
lines = f.readlines()
count = 0
for line in lines:
    count += 1
    json.loads(line)
    print("json load: " + line + " row = " + str(count))