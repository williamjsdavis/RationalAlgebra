# RationalAlgebra
Rational algebra operations for python

## Motivation

When solving linear algebra problems, it can be useful to use a computer to automate the calculations. However, Python's implementation of matrix operations often reverts to floating point representation of data. These incur a  small but noticeable error. 

<img width="589" alt="s1" src="https://user-images.githubusercontent.com/38541020/92814478-17d9db80-f378-11ea-850a-354fcf391834.png">

How can we do operations like this, but keep the results in the rational numbers? Use an algorithm that defines matrix operations purely using rational numbers. I based the algorithms used on the some from the [linear algebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/) package from `Julia`.

## Using the package: RationalAlgebra

The matrix is passed to `RationalAlgebra.RationalMatrix()` constructor, which instantiates it as a matrix of rational numbers.

Then, functions such as `RationalAlgebra.inv()` can be used to perform operations. The result is a matrix of rational numbers!

<img width="465" alt="s2" src="https://user-images.githubusercontent.com/38541020/92814951-9b93c800-f378-11ea-8a5c-33634bb4c012.png">

For more examples, see `example.ipynb`.

## Other features

Other features and operations that are available include:
- Row and column vectors of rational numbers.
- Operations between matricies and vectors (as well as other combinations).
  - Addition/subtraction
  - Scalar multiplication
  - Matrix multiplication
- LU decomposition
  - Method is [LUP decomposition](https://en.wikipedia.org/wiki/LU_decomposition), with partial piviting.
- Testing
  - Automatic testing is provided by the `test_basic.py` script.

## Future work

This was a small project, but in the future I want to add more features such as:
- Implement non-square matricies
- Implement getters and setters
- Implement convenience functions
    - Transpose
    - Matrix of zeros/ones
