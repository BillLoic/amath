# amath

This is a first python module with prime tools.

# Excamples

```python
>>> import amath
>>> n = 100
>>> amath.prime_factor(n)
[2, 5, 2, 5]
>>> list(amath.prime_list(20))
[2, 3, 5, 7, 11, 13, 17, 19]
>>> amath.div(1, 6, n=100)
0.166666666666666666666666666666666666666666666666666666666666666666667
>>> amath.summation(start=0, end=10, expression="2*x")
110
>>> import numpy as np
>>> import matplotlib.pyplot as plt
zeros = np.zeros(25)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'np' is not defined
>>> import numpy as np;zeros = np.zeros(25)
>>> import matplotlib.pyplot as plt
>>> list(range(26))
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
>>> len(list(range(26)))
26
>>> len(list(range(25)))
25
>>> rg=list(range(25))
>>> plt.plot(rg, prime, rg, zeros)
[<matplotlib.lines.Line2D object at 0x000001613E36B490>, <matplotlib.lines.Line2D object at 0x000001613E36B4F0>]
>>> plt.fill_between(rg, prime, zeros, alpha=0.25)
<matplotlib.collections.PolyCollection object at 0x000001613BFFBFD0>
>>> plt.show()
<Do it your self>
```
