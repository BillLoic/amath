# amath

This is a first python module with prime tools.

# Examples

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
>>> import numpy as np;zeros = np.zeros(25)
>>> import matplotlib.pyplot as plt
>>> prime = list(amath.prime_list(100))
>>> rg=list(range(25))
>>> plt.plot(rg, prime, rg, zeros)
[<matplotlib.lines.Line2D object at 0x000001613E36B490>, <matplotlib.lines.Line2D object at 0x000001613E36B4F0>]
>>> plt.fill_between(rg, prime, zeros, alpha=0.25)
<matplotlib.collections.PolyCollection object at 0x000001613BFFBFD0>
>>> plt.show()
<img src="https://raw.githubusercontent.com/BillLoic/amath/main/primeplot.png" decoding="async">
```
