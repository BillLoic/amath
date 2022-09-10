"""
# AMATH

# Why select this?
1. This module has more functions
2. No overflow problem
3. It supports more option than the basic calculate

# methods / classes
pi: 3.141592653589793
e: 2.718281828459045
I: 1j (such i)
mul: return the multiplication of a number
sqrt: return the square root of a number
square: return the square of a number (2D)
block: return the block of a number (3D)
isprime: check if a number is prime
prime_factor: the prime factor of a number
nPr: the number of permutations
nCr: the number of combinations
hypot: return the Euclidean distance
e_pow: the e power of a number
super_block: return the super block of a number (4D)
sin: return the sin of a number
cos: return the cos of a number
tan: return the tan of a number
cosh: return the cosh of a number
sinh: return the sinh of a number
tanh: return the tanh of a number
log: return the log of a number
div: return x / y
"""
from types import CodeType
from typing import SupportsIndex, Iterable
import math
import sympy as sp
import random
import sys
from time import time
import decimal
from sympy import (
    Rational, Integer
)
import warnings as _warn
from .error import *


_MAX = sys.float_info.max
_MIN = sys.float_info.min
INFINITY = float('inf')
NEGATIVE_INFINITY = float('-inf')
NOT_A_NUMBER = float('nan')
pi = 3.14159265358979323846
e = 2.71828182845904523536
phi = 1.61803398874989484820458683436563811772030917980576286213544862270526046281890
IMAGENARY_NUMBER = 1j
j = 1j
I = 1j

RECURSION_LIMIT = sys.getrecursionlimit()

def isnegative(num):
    '''Check if a number is negative'''
    return num < 0

def iszero(num):
    '''check if a number is zero'''
    return num == 0

def isone(num):
    """check if a number is one"""
    return num == 1
    
def isprime(num, timeout = False, second = 1e-2):
    '''Check if a number is prime'''
    startTime = time()
    if isnegative(num):
        raise MathError('Must be positive number')
    if iszero(num) or isone(num):
        return False
    try:
        for i in range(2, int(num**0.5)+1):
            if time() - startTime > second and timeout:
                raise TimeOutError('timeout')
            if num % i == 0:
                return False
        return True
    except KeyboardInterrupt:
        raise Terminate('isprime break')

def _prime_factor(num):
    '''The prime factor of a number'''
    _warn.warn("This function is just for calculate, call amath.prime_factor to return list.", category=CallSystemFunctionWarning)
    if num <= 0:
        raise ValueError('math domain error')
    if isprime(num):
        yield num
    try:
        for i in range(2, int(num**0.5)+1):
            if not isprime(i):
                continue
            if num % i == 0:
                yield i
                yield from _prime_factor(num // i)
                break
    except KeyboardInterrupt:
        raise Terminate('prime_factor break')
    except OverflowError:
        raise MathError('Number too large')
    except RecursionError:
        raise CalculateLimitExceededError('Maxinum recursion depth exceeded, Please set a higher value of RECURSION_LIMIT')
    
def prime_factor(num):
    return list(_prime_factor(num))

def prime_list(num):
    """
    Return a list of prime numbers (from 0 to num)
    """
    for i in range(1, num):
        if isprime(i):
            yield i
     
def infinity_primegen():
    """
    Generate infinity primes by generator.
    """
    n = 1
    while True:
        if isprime(n):
            yield n
        n += 1
            
def degrees(radians):
    '''Convert radians to degrees'''
    return radians * 180 / pi

def radians(degrees : float) -> float:
    '''
    Convert degrees to radians.
    '''
    return degrees * pi / 180

def fib(__n:int):
    '''The fibonacci sequence'''
    if __n < 0:
        raise ValueError('math domain error')
    if __n == 0:
        return 0
    if __n == 1:
        return 1
    try:
        return fib(__n - 1) + fib(__n - 2)
    except RecursionError:
        raise CalculateLimitExceededError('Maxinum recursion depth exceeded, Please set a higher value of RECURSION_LIMIT')

def fibrange(n):
    '''range of fibonacci numbers between 0 to n'''
    w = []
    for i in range(1,n):
        w.append(fib(i))
    return w
    
def div(x, y,n=30, allow_zero=False, allow_inf=False, rational=False) -> float:
    """
    such x/y
    allow_zero: if y is zero, return inf or nan, else, raise ZeroDivisionError.
    allow_inf: if x or y is inf, return 0, or nan, else, raise MathError.
    rational: if True, return Rational object, else return float.
    """
    if allow_zero:
        if allow_inf:
            if y == 0 and x == 0:
                return 0
            elif y==0:
                return INFINITY
            else:
                if not rational:
                    return sp.N(sp.S(x) / y, n=n)
                else:
                    return Rational(x / y)
        else:
            if x == INFINITY or y == INFINITY or x == NEGATIVE_INFINITY or y == NEGATIVE_INFINITY:
                raise MathError("Divide by infinity")
            else:
                try:
                    if not rational:
                        return sp.N(sp.S(x)/y, n=n)
                    else:
                        return Rational(x / y)
                except ZeroDivisionError: 
                    return INFINITY
                
    else:
        if allow_inf:
            return sp.N(sp.S(x) / y, n=n)
        else:
            if x == INFINITY or y == INFINITY or x == NEGATIVE_INFINITY or y == NEGATIVE_INFINITY:
                raise MathError("Divide by infinity")
            else:
                return sp.N(sp.S(x) / y, n=n)

def Pol(x, y):
    '''The polar coordinate of a point'''
    return math.hypot(x, y)

def Rec(r, n):
    '''The recursion of a number'''
    try:
        if n == 0:
            return 1
        else:
            return r * Rec(r, n - 1)
    except RecursionError:
        raise CalculateLimitExceededError('Maxinum recursion depth exceeded, Please set a higher value of RECURSION_LIMIT')

def nPr(n : int, r : int):
    '''The number of permutations
    n: the number of objects
    r: the number of objects to choose
    '''
    if n < r:
        raise ValueError('math domain error')
    if n <= 0 or r <= 0:
        return 0
    if n<0 or r<0:
        raise ValueError('math domain error')
    return math.factorial(n) // math.factorial(n - r)

def nCr(n : int, r : int):
    '''The number of combinations
    n: the number of objects
    r: the number of objects to choose
    '''
    if n < r:
        raise ValueError('math domain error')
    if n <= 0 or r <= 0:
        return 0
    if n < 0 or r < 0:
        raise ValueError('math domain error')
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))


def hypot(x : float, y : float) -> float:
    '''
    Return the Euclidean distance, sqrt(x**2 + y**2).
    '''
    return math.hypot(x, y)

def exp(num):
    '''The e power of a number'''
    if isinstance(num, complex):
        return cos(num) + 1j * sin(num)
    else:
        return math.e ** num

def ln(n):
    """
    Return a number of log (base=e)
    """
    return math.log(n, e)

def mul(args : Iterable[float | int | complex | object]):
    '''
    (The args's object must need to supported * operator, else raise TypeError)\n
    The multiplication of a number
    args: the number to multiply
    it will return the product of all the numbers
    >>> l = [1,5,6,6,4]
    >>> mul(l)
    720'''
    if len(args) == 0:
        return 0
    if len(args) == 1:
        return args[0]
    else:
        return args[0] * mul(args[1:])

def sqrt(__n: float|int):
    '''The square root of a number
    >>> sqrt(4) # \u221a4
    2.0
    >>> sqrt(-1) # \u221a-1
    1.0j
    >>> sqrt(1j) # \u221ai
    Traceback (most recent call last):
        ...snip...
    ValueError: math domain error
    '''
    if isinstance(__n, complex):
        raise ValueError('math domain error')
    if __n < 0:
        return math.sqrt(-__n) * 1j
    else:
        return math.sqrt(__n)

def square(num):
    '''The square of a number (2D)'''
    return num * num

def block(num):
    '''The block of a number (3D)'''
    return num * num * num

def super_block(num):
    '''The super block of a number (4D)'''
    return num * num * num * num

def sin(num):
    '''The sin of a number'''
    return math.sin(num)

def cos(num):
    '''The cos of a number'''
    return math.cos(num)

def tan(num):
    '''The tan of a number'''
    return math.tan(num)

def cosh(num):
    '''The cosh of a number'''
    return math.cosh(num)

def sinh(num):
    '''The sinh of a number'''
    return math.sinh(num)

def tanh(num):
    '''The tanh of a number'''
    return math.tanh(num)

def log(__x : float|int, base : float|int = 10) -> float:
    '''The log of a number'''
    if __x <= 0 or base <= 0:
        raise ValueError('math domain error')
    
    return math.log(__x, base)

def root(__n:int|float, __x:int|float = 3) -> float:
    '''The root of a number, default is 3
    like \u221b or \u221c in math
    >>> root(2,3) # \u221b9
    1.2599210498948732
    >>> root(16,4) # \u221c9
    2.0
    '''
    if __x <= 0:
        raise ValueError('math domain error')
    elif __x == 2:
        return sqrt(__n)
    return __n ** (1 / __x)

def factorial(__n : int):
    '''The factorial of a number'''
    if __n < 0:
        raise ValueError('math domain error')
    return math.factorial(__n)

def turn_base(__n : str, __base : int):
    '''Turn the number base to another base
    '''
    if __n < 0:
        raise ValueError('math domain error')
    try:
        return int(__n, __base)
    except ValueError:
        raise MathError(f'"{__n}" is not a base-{__base} number')

def summation(expression:str | CodeType, start=0, end:int|SupportsIndex=1000) -> int | float | complex:
    '''The summation of a expression
    expression ( f(x) ): the expression to calculate (raise MathError if expression is invaild)
    start: the start number
    end: the end number
    excample:
    >>> summation('4x', 0, 10) # (4*1) + (4*2) + (4*3) + (4*4) + (4*5) + (4*6) + (4*7) + (4*8) + (4*9) + (4*10)
    120
    '''
    step = 1
    answer = 0
    if start > end:
        step = -1
    if start == end:
        raise ValueError('math domain error')
    elif 'x' not in expression:
        raise MathError('The expression must contain x')

    for x in range(start, end + 1, step):
        try:
            answer += eval(expression)
        except OverflowError:
            raise MathError('The result is too large to be calculated')
        except SyntaxError:
            raise MathError('The expression is not valid')
        except NameError as error:
            raise ValueError(error, 'Please set the globals and locals keyword arguments')

    return answer

def product(expression:str | CodeType , start=1, end=1000,globals = globals(), locals = locals()) -> int | float | complex:
    '''The infinity product of a expression
    expression: the expression to calculate
    start: the start number
    end: the end number
    excample:
    >>> product('x', 1, 10) # 1*2*3*4*5*6*7*8*9*10
    3628800
    '''
    step = 1
    answer = 1
    if start == 0:
        raise ValueError('start can not be zero')
    if start > end:
        step = -1
        
    if start == end:
        raise ValueError('math domain error')
    elif 'x' not in expression:
        raise MathError('The expression must contain x')

    for x in range(start, end + 1, step):
        if x == 0:
            continue
        try:
            answer *= eval(expression,globals,locals)
        except OverflowError:
            raise MathError('The result is too large to be calculated')
        except SyntaxError:
            raise MathError('The expression is not valid')
        except NameError as error:
            raise ValueError(error, 'Please set the globals and locals keyword arguments')
    return answer

fact = factorial
base_transform = turn_base

    
class Decimal(decimal.Decimal):
    pass
    

class Number(float):
    def __init__(self, min : int | float , max: int | float):
        if min > max:
            raise ValueError("Min must be less than max")
        elif min.__class__ != max.__class__:
            raise TypeError("Min and max must be the same type")

        
        self.min = min
        self.max = max
    def __new__(cls, value):
        if isinstance(value, int):
            return int.__new__(cls, value)
        else:
            raise ValueError("Invalid value")

    def __str__(self):
        return "Number(value = {})".format(self)

    def __repr__(self):
        return "Number(value = {})".format(self)

    def __add__(self, other):
        return Number(self + other)

    def __sub__(self, other):
        return Number(self - other)

    def __mul__(self, other):
        return Number(self * other)

    def __truediv__(self, other):
        return Number(self / other)

    def __abs__(self):
        return Number(abs(self))

    def random(self):
        return Number(random.randint(self.min, self.max))

    def make_range(self, step):
        return range(self.min, self.max, step)

    def make_list(self, step):
        return list(range(self.min, self.max, step))

    def make_tuple(self, step):
        return tuple(range(self.min, self.max, step))

    def make_set(self, step):
        return set(range(self.min, self.max, step))

    def make_frozenset(self, step):
        return frozenset(range(self.min, self.max, step))

    def make_dict(self, step):
        return dict(zip(range(self.min, self.max, step) , range(self.min, self.max, step)))

class Circle():
    '''
    A class for representing a circle.
    benchmark pi = 3.141592653589793'''
    def __init__(self, radius = None, diameter = None, area = None, circumference = None):
        if radius is not None:
            self.radius = radius
        elif diameter is not None:
            self.radius = diameter / 2
        elif area is not None:
            self.radius = sqrt(area / pi)
        elif circumference is not None:
            self.radius = circumference / (2 * math.pi)
        else:
            raise ValueError("Invalid value")

    diameter = property(lambda self: 2 * self.radius)
    area = property(lambda self: math.pi * self.radius ** 2)
    circumference = property(lambda self: 2 * math.pi * self.radius)
    volume = property(lambda self: (4 / 3 * math.pi) * self.radius ** 3)
    surface_area = property(lambda self: (4 * math.pi) * self.radius ** 2)

    def __str__(self):
        return '''
        Circle(
        radius = {},
        diameter = {},
        area = {},
        circumference = {},
        volume = {},
        surface_area = {}
        )'''.format(self.radius, self.diameter, self.area, self.circumference, self.volume, self.surface_area)


class NumberRange(Number):
    def __init__(self, __min : int | float = 0 ,* , __max: int | float, __step: int | float = 1):
        if __min > __max:
            raise ValueError("Min must be less than max")
        elif __min.__class__ != __max.__class__:
            raise TypeError("Min and max must be the same type")
        
        self.__min = __min
        self.__max = __max
        self.__step = __step

    min = property(lambda self: self.__min)
    max = property(lambda self: self.__max)
    step = property(lambda self: self.__step)

    def __range__(self):
        try:
            return range(self.min, self.max, self.step)
        except:
            raise InvalidArgument('Invalid value')

    def __list__(self):
        try:
            return list(range(self.min, self.max, self.step))
        except:
            raise InvalidArgument('Invalid value')

    def __tuple__(self):
        try:
            return tuple(range(self.min, self.max, self.step))
        except:
            raise InvalidArgument('Invalid value')

    def __set__(self):
        try:
            return set(range(self.min, self.max, self.step))
        except:
            raise InvalidArgument('Invalid value')

    def __frozenset__(self):
        try:
            return frozenset(range(self.min, self.max, self.step))
        except:
            raise InvalidArgument('Invalid value')

    def random(self):
        return Number(random.randint(self.min, self.max))

    def sample(self, n):
        return random.sample(range(self.min, self.max, self.step), n)

    def shuffle(self):
        return random.shuffle(range(self.min, self.max, self.step))

    def choice(self):
        return random.choice(range(self.min, self.max, self.step))

class Rational(Rational):
    pass

class S(Integer):
    pass
    
if __name__ == "__main__":
    # print(Number(1, 10).make_list(2))
    # print(Number(1, 10).make_tuple(2))
    # print(Number(1, 10).make_set(2))
    # print(Number(1, 10).make_frozenset(2))
    # print(Number(1, 10).make_dict(2))
    # print(Number(1, 10).random())
    # print(Number(1, 10).make_range(2))
    # print(Number(1, 10).make_list(2))
    # print(Number(1, 10).make_tuple(2))
    # print(Number(1, 10).make_set(2))
    # print(Number(1, 10).make_frozenset(2))
    # print(Number(1, 10).make_dict(2))
    # print(Number(1, 10).random())
    # print(Number(1, 10).make_range(2))
    # print(Number(1, 10).make_list(2))
    # print(Number(1, 10).make_tuple(2))
    # print(Number(1, 10).make_set(2))
    # print(Number(1, 10).make_frozenset(2))
    # print(Number(1, 10).make_dict(2))
    # print(Number(1, 10).random())
    # print(Number(1, 10).make_range(2))
    # print(Number(1, 10).make_list(2))
    # print(Number(1, 10).make_tuple(2))
    # print(Number(1, 10).make_set(2))
    # print(Number(1, 10).make_frozenset(2))
    # print(Number(1, 10).make_dict(2))
    # print(Number(1, 10).random())
    # print(Number(1, 10).make_range(2))
    # print(Number(1, 10).make_list(2)) 
    pass
