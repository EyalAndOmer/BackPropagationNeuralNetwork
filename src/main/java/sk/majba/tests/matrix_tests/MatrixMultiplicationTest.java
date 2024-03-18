package sk.majba.tests.matrix_tests;

import org.junit.Test;

import static org.junit.Assert.*;
import static sk.majba.backpropagationneuralnetwork.be.utils.MatrixUtils.multiply;

public class MatrixMultiplicationTest {
    // 2D matrix with scalar
    @Test
    public void testMultiplicationWithPositiveScalar() {
        double[][] matrix = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        double scalar = 2.0f;
        double[][] expected = {{2.0f, 4.0f}, {6.0f, 8.0f}};
        assertArrayEquals(expected, multiply(matrix, scalar));
    }

    @Test
    public void testMultiplicationWithNegativeScalar() {
        double[][] matrix = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        double scalar = -1.0f;
        double[][] expected = {{-1.0f, -2.0f}, {-3.0f, -4.0f}};
        assertArrayEquals(expected, multiply(matrix, scalar));
    }

    @Test
    public void testMultiplicationWithZero() {
        double[][] matrix = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        double scalar = 0.0f;
        double[][] expected = {{0.0f, 0.0f}, {0.0f, 0.0f}};
        assertArrayEquals(expected, multiply(matrix, scalar));
    }

    // 2D matrix with 2D matrix
    @Test
    public void testMatrixMultiplicationPositive1() {
        double[][] matrix1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        double[][] matrix2 = {{5.0f, 6.0f}, {7.0f, 8.0f}};
        double[][] expected = {{19.0f, 22.0f}, {43.0f, 50.0f}};

        assertArrayEquals(expected, multiply(matrix1, matrix2));
    }

    @Test
    public void testValidMultiplication() {
        double[][] matrixA = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        double[][] matrixB = {{5.0f, 6.0f}, {7.0f, 8.0f}};
        double[][] expected = {{19.0f, 22.0f}, {43.0f, 50.0f}};
        assertArrayEquals(expected, multiply(matrixA, matrixB));
    }

    @Test
    public void testInvalidMultiplication() {
        double[][] matrixA = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        double[][] matrixB = {{5.0f, 6.0f}};
        assertThrows(IllegalArgumentException.class, () -> multiply(matrixA, matrixB));
    }

    @Test
    public void testMultiplicationWithZeroMatrix() {
        double[][] matrixA = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        double[][] matrixB = {{0.0f, 0.0f}, {0.0f, 0.0f}};
        double[][] expected = {{0.0f, 0.0f}, {0.0f, 0.0f}};
        assertArrayEquals(expected, multiply(matrixA, matrixB));
    }

    @Test
    public void testMultiplicationWithIdentityMatrix() {
        double[][] matrixA = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        double[][] matrixB = {{1.0f, 0.0f}, {0.0f, 1.0f}};
        assertArrayEquals(matrixA, multiply(matrixA, matrixB));
    }
}
