import fractions as _fractions
import numpy as _np
class RationalMatrix:
    # Only supports square matrices
    def __init__(self, intMatrix):
        _checkSquare(intMatrix)
        self.matrix = intMatrix + _fractions.Fraction()
        self.length = len(intMatrix)
    def __str__(self):
        max_chars = max([len(str(item)) for item in self.matrix.flat])
        formStr = '%0' + str(max_chars) + 's'
        strFun = lambda x: formStr % str(x)
        return '['+',\n '.join('['+', '.join(map(strFun, row))+
                             ']' for row in self.matrix)+']'
def inv(inRM):
    M = inRM.matrix
    A, ipiv = _LUrational(M)
    Mi = _invLU(A, ipiv)
    return RationalMatrix(Mi)
def _checkSquare(M):
    if _isSquare(M) is not True:
        raise ValueError('Not square matrix')
def _isSquare(M):
    return all(len(row) == len(M) for row in M)
def _setTriU(M, k=0):
    _checkSquare(M)
    n = len(M)
    U = _np.copy(M)
    for j in range(min(n, n+k-1)):
        for i in range(max(0, j-k+1), n):
            U[i,j] = _fractions.Fraction(0)
    return U
def _setTriL(M, k=0):
    _checkSquare(M)
    n = len(M)
    L = _np.copy(M)
    for j in range(min(0, k+2), n):
        for i in range(min(j-k, n)):
            L[i,j] = _fractions.Fraction(0)
    return L
def _setUnitDiag(M):
    _checkSquare(M)
    n = len(M)
    D = _np.copy(M)
    for i in range(n):
        D[i,i] = _fractions.Fraction(1)
    return D
def _isTriU(M, k=0):
    _checkSquare(M)
    n = len(M)
    for j in range(min(n, n+k-1)):
        for i in range(max(0, j-k+1), n):
            if M[i,j] != 0:
                return False
    return True
def _isTriL(M, k=0):
    _checkSquare(M)
    n = len(M)
    for j in range(min(0, k+2), n):
        for i in range(min(j-k, n)):
            if M[i,j] != 0:
                return False
    return True
def _reciprocal(A):
    return _fractions.Fraction(A._denominator, A._numerator)
def lu(inRM):
    M = inRM.matrix
    _checkSquare(M)
    A = _np.copy(M)
    n = len(A)
    ipiv = _np.zeros(n, dtype=int)
    info = 0
    for k in range(n):
        kp = k
        amax = _fractions.Fraction(0,1)
        for i in range(k, n):
            absi = abs(A[i,k])
            if absi > amax:
                kp = i
                amax = absi
        ipiv[k] = kp
        if A[kp,k] != 0:
            if k != kp:
                # Interchange
                for i in range(n):
                    tmp = A[k,i]
                    A[k,i] = A[kp,i]
                    A[kp,i] = tmp
            # Scale first column
            Akkinv = _reciprocal(A[k,k])
            for i in range(k+1,n):
                A[i,k] *= Akkinv
        elif info == 0:
            info = k
        for j in range(k+1, n):
            for i in range(k+1, n):
                A[i,j] -= A[i,k]*A[k,j]
    L = _setUnitDiag(_setTriL(A))
    U = _setTriU(A)
    return RationalMatrix(L), RationalMatrix(U), ipiv
def _LUrational(M):
    _checkSquare(M)
    A = _np.copy(M)
    n = len(A)
    ipiv = _np.zeros(n, dtype=int)
    info = 0
    for k in range(n):
        kp = k
        amax = _fractions.Fraction(0,1)
        for i in range(k, n):
            absi = abs(A[i,k])
            if absi > amax:
                kp = i
                amax = absi
        ipiv[k] = kp
        if A[kp,k] != 0:
            if k != kp:
                # Interchange
                for i in range(n):
                    tmp = A[k,i]
                    A[k,i] = A[kp,i]
                    A[kp,i] = tmp
            # Scale first column
            Akkinv = _reciprocal(A[k,k])
            for i in range(k+1,n):
                A[i,k] *= Akkinv
        elif info == 0:
            info = k
        for j in range(k+1, n):
            for i in range(k+1, n):
                A[i,j] -= A[i,k]*A[k,j]
    return A, ipiv
def _applyIpivRows(A, ipiv):
    n = len(A)
    B = _np.identity(n).astype(int) + _fractions.Fraction()
    for i, j in enumerate(ipiv):
        if i != j:
            for col in range(n):
                B[i,col], B[j,col] = B[j,col], B[i,col]
    return B
def _ldivLU(A, B, Mtype):
    _checkSquare(A)
    nA = len(A)
    tmp = _np.zeros(nA).astype(int) + _fractions.Fraction()
    nB = len(B)
    for i in range(nB):
        _unsafeCopyTo(tmp, 0, B, i*nB, nB)
        if Mtype is 'L':
            _naivesubL(A, tmp)
        elif Mtype is 'U':
            _naivesubU(A, tmp)
        _unsafeCopyTo(B, i*nB, tmp, 0, nB)
    return B.transpose()
def _naivesubL(A, b):
    n = len(A)
    x = _np.copy(b)
    for j in range(n):
        x[j] = b[j]
        xj = x[j]
        for i in range(j+1,n):
            b[i] -= A[i,j]*xj
    return x
def _naivesubU(A, b):
    n = len(A)
    x = b
    for j in reversed(range(n)):
        x[j] = b[j] / A[j,j]
        xj = x[j]
        for i in reversed(range(0,j)):
            b[i] -= A[i,j]*xj
    return x
def _unsafeCopyTo(dest, destI, src, srcI, N):
    for n in range(N):
        dest.flat[destI+n] = src.flat[srcI+n]
def _invLU(M, ipiv):
    B = _applyIpivRows(M, ipiv)
    L = _setUnitDiag(_setTriL(M))
    U = _setTriU(M)
    tmp = _ldivLU(L, B.transpose(), 'L')
    return _ldivLU(U, tmp.transpose(), 'U')