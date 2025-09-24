from functools import partial

def funct(a = 4,b = 10,c = 10):

    return (a+b)*c

bar = partial(
    funct,
    b = 3,
    c = 1
)

bar2 = partial(
    funct,
    a = 3,
    c = 2
)
d = bar(a=1)
d2 = bar2(b=2)

print(d2)