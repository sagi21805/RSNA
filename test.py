import time

x = 0
t = time.time()
for i in range(1000000):
    for j in range(1000000):
        x += i *j
print(time.time() - t)
print(x)