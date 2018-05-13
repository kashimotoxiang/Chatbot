import functools


class A():
    def __init__(self, x):
        A.x = x

    def aa(self, y, z):
        print('aa,{}'.format(self.x + y + z))

    def bb(self):
        return 'bb'


fun = A.aa
fun
model = A(1)
a = fun(model, 2, 3)

fun1 = functools.partial(fun, z=1)
fun1(model, y=1)
