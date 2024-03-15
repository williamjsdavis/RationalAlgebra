import sys
sys.path.insert(1, '../RationalAlgebra')

import unittest
import numpy as np
import RationalAlgebra.RationalAlgebra as ra
from fractions import Fraction as Fr
from operator import matmul

class TestAssignMethods(unittest.TestCase):
    def setUp(self):
        self.exMatrix = ra.RationalMatrix((10*np.random.rand(5,5)).astype(int))
        self.exRowVector = ra.RationalVector((10*np.random.rand(1,5)).astype(int))
        self.exColumnVector = ra.RationalVector((10*np.random.rand(5,1)).astype(int))
    def test_assign_matrix(self):
        self.assertEqual(self.exMatrix.value.shape, (5, 5))
        self.assertEqual(type(self.exMatrix.value[0,0]), Fr)
        self.assertEqual(type(self.exMatrix), ra.RationalMatrix)
    def test_assign_row_vector(self):
        self.assertEqual(self.exRowVector.value.shape, (1, 5))
        self.assertEqual(type(self.exRowVector.value[0,0]), Fr)
        self.assertEqual(type(self.exRowVector), ra.RationalVector)
    def test_assign_row_vector(self):
        self.assertEqual(self.exColumnVector.value.shape, (5, 1))
        self.assertEqual(type(self.exColumnVector.value[0,0]), Fr)
        self.assertEqual(type(self.exColumnVector), ra.RationalVector)
        
class TestAddMatrixMethods(unittest.TestCase):
    def setUp(self):
        self.exMatrix = ra.RationalMatrix((10*np.random.rand(5,5)).astype(int))
        self.exMatrix2 = ra.RationalMatrix((10*np.random.rand(5,5)).astype(int))
        self.matAdd1 = self.exMatrix + self.exMatrix2
        self.matAdd2 = self.exMatrix2 + self.exMatrix
        self.intAdd1 = 5 + self.exMatrix
        self.intAdd2 = self.exMatrix + 5
        self.fracAdd1 = Fr(1,3) + self.exMatrix
        self.fracAdd2 = self.exMatrix + Fr(1,3)
        self.floatAdd1 = 0.5 + self.exMatrix
        self.floatAdd2 = self.exMatrix + 0.5
    def test_equality(self):
        myEqual = lambda x,y: all(x.value.flat == y.value.flat)
        self.assertTrue(myEqual(self.intAdd1, self.intAdd2))
        self.assertTrue(myEqual(self.fracAdd1, self.fracAdd2))
        self.assertTrue(myEqual(self.floatAdd1, self.floatAdd2))
    def test_size(self):
        self.assertEqual(self.intAdd1.value.shape, (5, 5))
        self.assertEqual(self.fracAdd1.value.shape, (5, 5))
        self.assertEqual(self.floatAdd1.value.shape, (5, 5))
    def test_element_type(self):
        self.assertEqual(type(self.intAdd1.value[0,0]), Fr)
        self.assertEqual(type(self.fracAdd1.value[0,0]), Fr)
        self.assertEqual(type(self.floatAdd1.value[0,0]), Fr)
    def test_type(self):
        self.assertEqual(type(self.intAdd1), ra.RationalMatrix)
        self.assertEqual(type(self.fracAdd1), ra.RationalMatrix)
        self.assertEqual(type(self.floatAdd1), ra.RationalMatrix)

class TestSubMatrixMethods(unittest.TestCase):
    def setUp(self):
        self.exMatrix = ra.RationalMatrix(np.zeros((5,5)).astype(int))
        self.exMatrix.value[2,4] = Fr(11,7)
        self.exMatrix.value[1,2] = Fr(3,5)
        self.exMatrix2 = ra.RationalMatrix(np.zeros((5,5)).astype(int))
        self.exMatrix2.value[2,4] = Fr(1,7)
        self.exMatrix2.value[1,2] = Fr(4,7)
        
        self.exColumnVector = ra.RationalVector(np.zeros((5,1)).astype(int))
        self.exColumnVector.value[2] = Fr(3,4)
        self.exColumnVector.value[4] = Fr(9,2)
        self.exColumnVector2 = ra.RationalVector(np.zeros((5,1)).astype(int))
        self.exColumnVector2.value[2] = Fr(1,4)
        self.exColumnVector2.value[4] = Fr(23,2)
        
        self.exRowVector = ra.RationalVector(np.zeros((1,5)).astype(int))
        self.exRowVector.value[0][2] = Fr(5,7)
        self.exRowVector.value[0][0] = Fr(9,2)
        self.exRowVector2 = ra.RationalVector(np.zeros((1,5)).astype(int))
        self.exRowVector2.value[0][2] = Fr(1,7)
        self.exRowVector2.value[0][0] = Fr(13,2)
        
        self.subMM = self.exMatrix - self.exMatrix2
        self.subCC = self.exColumnVector - self.exColumnVector2
        self.subRR = self.exRowVector - self.exRowVector2
    def test_shape(self):
        self.assertEqual(self.subMM.value.shape, (5, 5))
        self.assertEqual(self.subCC.value.shape, (5, 1))
        self.assertEqual(self.subRR.value.shape, (1, 5))
    def test_value(self):
        self.assertEqual(self.subMM.value[0,0], Fr(0,1))
        self.assertEqual(self.subMM.value[2,4], Fr(10,7))
        self.assertEqual(self.subMM.value[1,2], Fr(1,35))
        
        self.assertEqual(self.subCC.value[0], Fr(0,1))
        self.assertEqual(self.subCC.value[2], Fr(1,2))
        self.assertEqual(self.subCC.value[4], Fr(-7,1))
    
        self.assertEqual(self.subRR.value[0,1], Fr(0,1))
        self.assertEqual(self.subRR.value[0,2], Fr(4,7))
        self.assertEqual(self.subRR.value[0,0], Fr(-2,1))
    def test_element_type(self):
        self.assertEqual(type(self.subMM.value[1,4]), Fr)
        self.assertEqual(type(self.subCC.value[0][0]), Fr)
        self.assertEqual(type(self.subRR.value[0,0]), Fr)
    def test_type(self):
        self.assertEqual(type(self.subMM), ra.RationalMatrix)
        self.assertEqual(type(self.subCC), ra.RationalVector)
        self.assertEqual(type(self.subRR), ra.RationalVector)
        
class TestMultiplyMatrixMethods(unittest.TestCase):
    def setUp(self):
        self.exMatrix = ra.RationalMatrix((10*np.random.rand(5,5)).astype(int))
        self.intMultiply1 = 5 * self.exMatrix
        self.intMultiply2 = self.exMatrix * 5
        self.fracMultiply1 = Fr(1,3) * self.exMatrix
        self.fracMultiply2 = self.exMatrix * Fr(1,3)
        self.floatMultiply1 = 0.5 * self.exMatrix
        self.floatMultiply2 = self.exMatrix * 0.5
    def test_equality(self):
        myEqual = lambda x,y: all(x.value.flat == y.value.flat)
        self.assertTrue(myEqual(self.intMultiply1, self.intMultiply2))
        self.assertTrue(myEqual(self.fracMultiply1, self.fracMultiply2))
        self.assertTrue(myEqual(self.floatMultiply1, self.floatMultiply2))
    def test_size(self):
        self.assertEqual(self.intMultiply1.value.shape, (5, 5))
        self.assertEqual(self.fracMultiply1.value.shape, (5, 5))
        self.assertEqual(self.floatMultiply1.value.shape, (5, 5))
    def test_element_type(self):
        self.assertEqual(type(self.intMultiply1.value[0,0]), Fr)
        self.assertEqual(type(self.fracMultiply1.value[0,0]), Fr)
        self.assertEqual(type(self.floatMultiply1.value[0,0]), Fr)
    def test_type(self):
        self.assertEqual(type(self.intMultiply1), ra.RationalMatrix)
        self.assertEqual(type(self.fracMultiply1), ra.RationalMatrix)
        self.assertEqual(type(self.floatMultiply1), ra.RationalMatrix)
        
class TestAddVectorMethods(unittest.TestCase):
    def setUp(self):
        self.exRowVector = ra.RationalVector((10*np.random.rand(1,5)).astype(int))
        self.exRowVector2 = ra.RationalVector((10*np.random.rand(1,5)).astype(int))
        self.exColumnVector = ra.RationalVector((10*np.random.rand(5,1)).astype(int))
        self.exColumnVector2 = ra.RationalVector((10*np.random.rand(5,1)).astype(int))
        
        self.rowVecAdd1 = self.exRowVector + self.exRowVector2
        self.rowVecAdd2 = self.exRowVector2 + self.exRowVector
        self.colVecAdd1 = self.exColumnVector + self.exColumnVector2
        self.colVecAdd2 = self.exColumnVector2 + self.exColumnVector
        self.intRowAdd1 = 5 + self.exRowVector
        self.intRowAdd2 = self.exRowVector + 5
        self.fracRowAdd1 = Fr(1,3) + self.exRowVector
        self.fracRowAdd2 = self.exRowVector + Fr(1,3)
        self.floatRowAdd1 = 0.5 + self.exRowVector
        self.floatRowAdd2 = self.exRowVector + 0.5
        self.intColAdd1 = 5 + self.exColumnVector
        self.intColAdd2 = self.exColumnVector + 5
        self.fracColAdd1 = Fr(1,3) + self.exColumnVector
        self.fracColAdd2 = self.exColumnVector + Fr(1,3)
        self.floatColAdd1 = 0.5 + self.exColumnVector
        self.floatColAdd2 = self.exColumnVector + 0.5
    def test_equality(self):
        myEqual = lambda x,y: all(x.value.flat == y.value.flat)
        self.assertTrue(myEqual(self.rowVecAdd1, self.rowVecAdd2))
        self.assertTrue(myEqual(self.colVecAdd1, self.colVecAdd2))
        self.assertTrue(myEqual(self.intRowAdd1, self.intRowAdd2))
        self.assertTrue(myEqual(self.fracRowAdd1, self.fracRowAdd2))
        self.assertTrue(myEqual(self.floatRowAdd1, self.floatRowAdd2))
        self.assertTrue(myEqual(self.intColAdd1, self.intColAdd2))
        self.assertTrue(myEqual(self.fracColAdd1, self.fracColAdd2))
        self.assertTrue(myEqual(self.floatColAdd1, self.floatColAdd2))
    def test_size(self):
        self.assertEqual(self.rowVecAdd1.value.shape, (1, 5))
        self.assertEqual(self.colVecAdd1.value.shape, (5, 1))
        self.assertEqual(self.intRowAdd1.value.shape, (1, 5))
        self.assertEqual(self.intColAdd1.value.shape, (5, 1))
        self.assertEqual(self.fracRowAdd1.value.shape, (1, 5))
        self.assertEqual(self.fracColAdd1.value.shape, (5, 1))
        self.assertEqual(self.floatRowAdd1.value.shape, (1, 5))
        self.assertEqual(self.floatColAdd1.value.shape, (5, 1))
    def test_element_type(self):
        self.assertEqual(type(self.rowVecAdd1.value[0,0]), Fr)
        self.assertEqual(type(self.colVecAdd1.value[0,0]), Fr)
        self.assertEqual(type(self.intRowAdd1.value[0,0]), Fr)
        self.assertEqual(type(self.intColAdd1.value[0,0]), Fr)
        self.assertEqual(type(self.fracRowAdd1.value[0,0]), Fr)
        self.assertEqual(type(self.fracColAdd1.value[0,0]), Fr)
        self.assertEqual(type(self.floatRowAdd1.value[0,0]), Fr)
        self.assertEqual(type(self.floatColAdd1.value[0,0]), Fr)
    def test_type(self):
        self.assertEqual(type(self.rowVecAdd1), ra.RationalVector)
        self.assertEqual(type(self.colVecAdd1), ra.RationalVector)
        self.assertEqual(type(self.intRowAdd1), ra.RationalVector)
        self.assertEqual(type(self.intColAdd1), ra.RationalVector)
        self.assertEqual(type(self.fracRowAdd1), ra.RationalVector)
        self.assertEqual(type(self.fracColAdd1), ra.RationalVector)
        self.assertEqual(type(self.floatRowAdd1), ra.RationalVector)
        self.assertEqual(type(self.floatColAdd1), ra.RationalVector)
        
class TestMultiplyRowVectorMethods(unittest.TestCase):
    def setUp(self):
        self.exRowVector = ra.RationalVector((10*np.random.rand(1,5)).astype(int))
        self.intMultiply1 = 5 * self.exRowVector
        self.intMultiply2 = self.exRowVector * 5
        self.fracMultiply1 = Fr(1,3) * self.exRowVector
        self.fracMultiply2 = self.exRowVector * Fr(1,3)
        self.floatMultiply1 = 0.5 * self.exRowVector
        self.floatMultiply2 = self.exRowVector * 0.5
    def test_equality(self):
        myEqual = lambda x,y: all(x.value.flat == y.value.flat)
        self.assertTrue(myEqual(self.intMultiply1, self.intMultiply2))
        self.assertTrue(myEqual(self.fracMultiply1, self.fracMultiply2))
        self.assertTrue(myEqual(self.floatMultiply1, self.floatMultiply2))
    def test_size(self):
        self.assertEqual(self.intMultiply1.value.shape, (1, 5))
        self.assertEqual(self.fracMultiply1.value.shape, (1, 5))
        self.assertEqual(self.floatMultiply1.value.shape, (1, 5))
    def test_element_type(self):
        self.assertEqual(type(self.intMultiply1.value[0,0]), Fr)
        self.assertEqual(type(self.fracMultiply1.value[0,0]), Fr)
        self.assertEqual(type(self.floatMultiply1.value[0,0]), Fr)
    def test_type(self):
        self.assertEqual(type(self.intMultiply1), ra.RationalVector)
        self.assertEqual(type(self.fracMultiply1), ra.RationalVector)
        self.assertEqual(type(self.floatMultiply1), ra.RationalVector)
        
class TestMultiplyColumnVectorMethods(unittest.TestCase):
    def setUp(self):
        self.exColumnVector = ra.RationalVector((10*np.random.rand(5,1)).astype(int))
        self.intMultiply1 = 5 * self.exColumnVector
        self.intMultiply2 = self.exColumnVector * 5
        self.fracMultiply1 = Fr(1,3) * self.exColumnVector
        self.fracMultiply2 = self.exColumnVector * Fr(1,3)
        self.floatMultiply1 = 0.5 * self.exColumnVector
        self.floatMultiply2 = self.exColumnVector * 0.5
    def test_equality(self):
        myEqual = lambda x,y: all(x.value.flat == y.value.flat)
        self.assertTrue(myEqual(self.intMultiply1, self.intMultiply2))
        self.assertTrue(myEqual(self.fracMultiply1, self.fracMultiply2))
        self.assertTrue(myEqual(self.floatMultiply1, self.floatMultiply2))
    def test_size(self):
        self.assertEqual(self.intMultiply1.value.shape, (5, 1))
        self.assertEqual(self.fracMultiply1.value.shape, (5, 1))
        self.assertEqual(self.floatMultiply1.value.shape, (5, 1))
    def test_element_type(self):
        self.assertEqual(type(self.intMultiply1.value[0,0]), Fr)
        self.assertEqual(type(self.fracMultiply1.value[0,0]), Fr)
        self.assertEqual(type(self.floatMultiply1.value[0,0]), Fr)
    def test_type(self):
        self.assertEqual(type(self.intMultiply1), ra.RationalVector)
        self.assertEqual(type(self.fracMultiply1), ra.RationalVector)
        self.assertEqual(type(self.floatMultiply1), ra.RationalVector)

class TestMatmulMethods(unittest.TestCase):
    def setUp(self):
        self.exMatrix = ra.RationalMatrix(np.zeros((5,5)).astype(int))
        self.exMatrix.value[2,4] = Fr(11,7)
        self.exMatrix.value[1,2] = Fr(3,5)
        
        self.exColumnVector = ra.RationalVector(np.zeros((5,1)).astype(int))
        self.exColumnVector.value[2] = Fr(3,4)
        self.exColumnVector.value[4] = Fr(9,2)
        
        self.exRowVector = ra.RationalVector(np.zeros((1,5)).astype(int))
        self.exRowVector.value[0][2] = Fr(5,7)
        self.exRowVector.value[0][0] = Fr(9,2)
        
        self.mulMM = self.exMatrix @ self.exMatrix
        self.mulMC = self.exMatrix @ self.exColumnVector
        self.mulCR = self.exColumnVector @ self.exRowVector
        self.mulRM = self.exRowVector @ self.exMatrix
        self.mulRC = self.exRowVector @ self.exColumnVector
    def test_shape(self):
        self.assertEqual(self.mulMM.value.shape, (5, 5))
        self.assertEqual(self.mulMC.value.shape, (5, 1))
        self.assertEqual(self.mulCR.value.shape, (5, 5))
        self.assertEqual(self.mulRM.value.shape, (1, 5))
        self.assertEqual(self.mulRC.value.shape, (1, 1))
    def test_value(self):
        self.assertEqual(self.mulMM.value[1,4], Fr(33,35))
        self.assertEqual(self.mulMM.value[1,3], Fr(0))
        self.assertEqual(self.mulMC.value[1], Fr(9,20))
        self.assertEqual(self.mulMC.value[3], Fr(0))
        self.assertEqual(self.mulCR.value[2,2], Fr(15,28))
        self.assertEqual(self.mulCR.value[2,3], Fr(0))
        self.assertEqual(self.mulRM.value[0][4], Fr(55,49))
        self.assertEqual(self.mulRM.value[0][3], Fr(0))
        self.assertEqual(self.mulRC.value[0], Fr(15,28))
    def test_element_type(self):
        self.assertEqual(type(self.mulMM.value[1,4]), Fr)
        self.assertEqual(type(self.mulMC.value[1][0]), Fr)
        self.assertEqual(type(self.mulCR.value[2,2]), Fr)
        self.assertEqual(type(self.mulRM.value[0][4]), Fr)
        self.assertEqual(type(self.mulRC.value[0][0]), Fr)
    def test_type(self):
        self.assertEqual(type(self.mulMM), ra.RationalMatrix)
        self.assertEqual(type(self.mulMC), ra.RationalVector)
        self.assertEqual(type(self.mulCR), ra.RationalMatrix)
        self.assertEqual(type(self.mulRM), ra.RationalVector)
        self.assertEqual(type(self.mulRC), ra.RationalVector)

class TestMatmulFailMethods(unittest.TestCase):
    def setUp(self):
        self.exMatrix = ra.RationalMatrix(np.zeros((5,5)).astype(int))
        self.exColumnVector = ra.RationalVector(np.zeros((5,1)).astype(int))
        self.exRowVector = ra.RationalVector(np.zeros((1,5)).astype(int))
    def test_fail_mul(self):
        self.assertRaises(ValueError, matmul, self.exMatrix, self.exRowVector)
        self.assertRaises(ValueError, matmul, self.exColumnVector, self.exMatrix)
        self.assertRaises(ValueError, matmul, self.exColumnVector, self.exColumnVector)
        self.assertRaises(ValueError, matmul, self.exRowVector, self.exRowVector)
        
class TestMatmulProperties(unittest.TestCase):
    def setUp(self):
        self.exMatrixA = ra.RationalMatrix((10*np.random.rand(5,5)).astype(int))
        self.exMatrixB = ra.RationalMatrix((10*np.random.rand(5,5)).astype(int))
        self.exMatrixC = ra.RationalMatrix((10*np.random.rand(5,5)).astype(int))
        self.exMatrixD = ra.RationalMatrix((10*np.random.rand(5,5)).astype(int))
    def test_associative(self):
        AB_C = (self.exMatrixA @ self.exMatrixB) @ self.exMatrixC
        A_BC = self.exMatrixA @ (self.exMatrixB @ self.exMatrixC) 
        myEqual = lambda x,y: all(x.value.flat == y.value.flat)
        self.assertTrue(myEqual(AB_C, A_BC))
    def test_distributive(self):
        A_BC = self.exMatrixA @ (self.exMatrixB + self.exMatrixC)
        AB_AC = (self.exMatrixA @ self.exMatrixB) + (self.exMatrixA @ self.exMatrixC)
        BC_D = (self.exMatrixB + self.exMatrixC) @ self.exMatrixD
        BD_CD = (self.exMatrixB @ self.exMatrixD) + (self.exMatrixC @ self.exMatrixD)
        myEqual = lambda x,y: all(x.value.flat == y.value.flat)
        self.assertTrue(myEqual(A_BC, AB_AC))
        self.assertTrue(myEqual(BC_D, BD_CD))
        
class TestInverseMethods(unittest.TestCase):
    def setUp(self):
        self.exMatrix = ra.RationalMatrix(np.identity(5).astype(int))
        self.exMatrix.value[2,4] = Fr(11,7)
        self.exMatrix.value[1,2] = Fr(3,5)
        self.exMatrix.value[4,2] = Fr(71,3)
    def test_inverse(self):
        invM = ra.inv(self.exMatrix)
        self.assertEqual(invM.value[1,2], Fr(63, 3800))
        self.assertEqual(invM.value[1,4], Fr(-99, 3800))
        self.assertEqual(invM.value[3,3], Fr(1))
        self.assertEqual(invM.value[0,3], Fr(0))
        self.assertEqual(type(invM.value[1,4]), Fr)
        self.assertEqual(type(invM), ra.RationalMatrix)
        self.assertEqual(invM.value.shape, (5, 5))
    def test_inverse_matmul(self):
        invM = ra.inv(self.exMatrix)
        I1 = invM @ self.exMatrix
        I2 = self.exMatrix @ invM
        
        myEqual = lambda x,y: all(x.value.flat == y.value.flat)
        self.assertTrue(myEqual(I1, I2))
        self.assertEqual(I1.value[2,2], Fr(1))
        self.assertEqual(I1.value[1,2], Fr(0))
        self.assertEqual(type(I1.value[1,4]), Fr)
        self.assertEqual(type(I1), ra.RationalMatrix)
        self.assertEqual(I1.value.shape, (5, 5))
    def test_lu_decomposition(self):
        L, U, P = ra.lu(self.exMatrix)
        self.assertEqual(L.value[4,2], Fr(3, 71))
        self.assertEqual(L.value[2,4], Fr(0))
        self.assertEqual(U.value[4,4], Fr(760, 497))
        self.assertEqual(U.value[2,4], Fr(1))
        self.assertEqual(U.value[2,1], Fr(0))
        self.assertEqual(P.value[4,2], Fr(1))
        self.assertEqual(P.value[4,3], Fr(0))
        
        self.assertEqual(type(L.value[1,4]), Fr)
        self.assertEqual(type(L), ra.RationalMatrix)
        self.assertEqual(L.value.shape, (5, 5))
        self.assertEqual(type(U.value[1,4]), Fr)
        self.assertEqual(type(U), ra.RationalMatrix)
        self.assertEqual(U.value.shape, (5, 5))
        self.assertEqual(type(P.value[1,4]), Fr)
        self.assertEqual(type(P), ra.RationalMatrix)
        self.assertEqual(P.value.shape, (5, 5))
def test_lu_matmul(self):
        L, U, P = ra.lu(self.exMatrix)
        PM = P @ self.exMatrix
        LU = L @ U
        PiLU = ra.inv(P) @ L @ U
        
        myEqual = lambda x,y: all(x.value.flat == y.value.flat)
        self.assertTrue(myEqual(PM, LU))
        self.assertTrue(myEqual(self.exMatrix, PiLU))
        
if __name__ == '__main__':
    unittest.main()
