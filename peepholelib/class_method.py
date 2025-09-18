
class foo():
    def __init__(self, x=0):
        self.a = x 

    @classmethod
    def cat(cls, insts):
        ret = cls()
        ret.a = []
        for inst in insts:
            ret.a.append(inst.a)
        return ret

if __name__ == '__main__':
    f1 = foo(2)
    f2 = foo(3)
    f3 = foo(4)

    r = foo.cat([f1, f2, f3])
    print('r: ', r.a)
