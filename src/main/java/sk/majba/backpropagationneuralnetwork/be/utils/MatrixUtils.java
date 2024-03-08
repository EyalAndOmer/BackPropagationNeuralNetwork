package sk.majba.backpropagationneuralnetwork.be.utils;

public class MatrixUtils {

    public static double[][] multiply(double[][] matrix, double scalar) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = matrix[i][j] * scalar;
            }
        }

        return result;
    }

    public static double[][] multiply(double[][] matrix1, double[][] matrix2) {
        if (matrix1 == null || matrix2 == null) {
            throw new NullPointerException();
        }

        if (matrix1.length == 0 || matrix2.length == 0 || matrix1[0].length != matrix2.length) {
            throw new IllegalArgumentException(String.format("Invalid matrices for multiplication, first matrix %d:%d," +
                    "second matrix %d:%d", matrix1.length, matrix1[0].length, matrix2.length, matrix2[0].length));
        }

        double[][] c = new double[matrix1.length][matrix2[0].length];

        for (int i = 0; i < matrix1.length; i++) {
            for (int j = 0; j < matrix2[0].length; j++) {
                for (int k = 0; k < matrix1[0].length; k++) {
                    c[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
        return c;
    }

    public static double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] transposedMatrix = new double[cols][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposedMatrix[j][i] = matrix[i][j];
            }
        }

        return transposedMatrix;
    }

}
