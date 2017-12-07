class Memoize:
    '''
    Use it as decorator @Memoize to compute a 
    function output once and no more
    
    Example:
    
    @Memoize
    def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)
    '''
    def __init__(self, f):
        self.f = f
        self.mem = {}
    def __call__(self, *args):
        if not args in self.mem:
            self.mem[args] = self.f(*args)
        return self.mem[args]