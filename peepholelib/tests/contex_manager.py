from numpy.random import randint
from functools import partial

class Foo():
    def __init__(self):
        self.a = randint(0,10) 
        self.close = partial(self.__exit__, None, None, None)
    
    def bar(self):
        print('bar: ', self.a)

    def __enter__(self):
        print('hi: ', self.a)
        return self

    def __exit__(self, a, b, c):
        print('adios: ', self.a)

if __name__ == '__main__':
    with Foo() as foo:
        print('aaaa')
        foo.bar()
        print('bbb')
    
    input('wait')

    foo = Foo()
    foo.bar()
    foo.close()
    
    foo2 = Foo()
    input('wait')
    with foo as f, foo2 as f2:
        f.bar()
        f2.bar()
