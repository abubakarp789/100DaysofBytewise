num = int(input("How many terms?: "))


print("Fibanocci Sequence: ")
print(x)
print(y)
for i in range(num):
    c = x + y
    x = y
    y = c
    print(c, end = " ")