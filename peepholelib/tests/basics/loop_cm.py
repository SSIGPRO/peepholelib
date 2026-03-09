from contextlib import ExitStack

class foo():
    def __init__(self, a):
        self.a = a
        self.b = None
    def __enter__(self):
        print('enter: ', self.a)

        self.b = self.a*2

    def __exit__(self, z, x, c):
        self.b = None
        print('exit: ', self.a)

    def bar(self):
        print('aaaa: ', self.a)
        print('bbbb: ', self.b)


if __name__ == "__main__":
    l = {'a': foo(3), 'b': foo(5), 'c': foo(6)}

    with ExitStack() as stack:
        c = [stack.enter_context(ll) for ll in l.values()]
        f = foo(7)
        stack.enter_context(f)

        for ll in l.values():
            ll.bar()
        f.bar()

    for ll in l.values():
        ll.bar()
    f.bar()
