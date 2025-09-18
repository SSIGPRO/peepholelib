import abc  

class foo():
    def __init__(self, x=0):
        self.a = x 
        return
    
    @abc.abstractmethod
    def bar(self, y):
        raise NotImplementedError()

    @classmethod
    def cat(cls, insts):
        ret = cls()
        ret.a = []
        for inst in insts:
            ret.a.append(inst.a)
        return ret

class foo2(foo):
    def __init__(self, x=0):
        foo.__init__(self, x)
        return

    def bar(self, y):
        self.a += y
        return

class foo3(foo):
    def __init__(self, x=0):
        foo.__init__(self, x)
        return

if __name__ == '__main__':
    f1 = foo2(2)
    f2 = foo2(3)
    f3 = foo3(4)

    f2.bar(10)

    r = foo.cat([f1, f2, f3])
    print('r: ', r.a)
