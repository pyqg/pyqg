from pyqg import delegate

def test_delegate():
    @delegate(*['e', 'f'], to='b')
    @delegate(*['g'], to='c', prefix='_')
    @delegate(*['h'], to='c', prefix='__')
    class A:
        def __init__(self, b, c):
            self.b = b
            self.c = c
            self.d = 'foo'

    class B:
        def __init__(self):
            self.e = 'bar'
            self.f = 'baz'

    class C:
        def __init__(self):
            self.e = 'NOPE'
            self.g = 'bat'
            self.h = 'bart'
    
    delegates = [('e', 'b', ''), ('f', 'b', ''),
                 ('g', 'c', '_'), ('h', 'c', '__')]

    assert A._delegates == delegates

    a = A(B(), C())

    assert a.d == 'foo'
    assert a.e == 'bar'
    assert a.f == 'baz'
    assert a._g == 'bat'
    assert a.__h == 'bart'

if __name__ == '__main__':
    test_delegate()
