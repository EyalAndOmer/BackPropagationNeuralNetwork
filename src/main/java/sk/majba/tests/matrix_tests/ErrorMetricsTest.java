package sk.majba.tests.matrix_tests;

import org.junit.Test;
import sk.majba.backpropagationneuralnetwork.be.error_metrics.MAPE;
import sk.majba.backpropagationneuralnetwork.be.error_metrics.MSE;
import sk.majba.backpropagationneuralnetwork.be.error_metrics.RMSE;

import static org.junit.Assert.assertEquals;

public class ErrorMetricsTest {
    private static final double DELTA = 1e-15f;

    @Test
    public void testMSE() {
        MSE mse = new MSE();
        double[] actual = {1.0f, 2.0f, 3.0f};
        double[] predicted = {1.0f, 2.0f, 3.0f};
        assertEquals(0.0f, mse.calculateError(actual, predicted), DELTA);
    }

    @Test
    public void testRMSE() {
        RMSE rmse = new RMSE();
        double[] actual = {1.0f, 2.0f, 3.0f};
        double[] predicted = {1.0f, 2.0f, 3.0f};
        assertEquals(0.0f, rmse.calculateError(actual, predicted), DELTA);
    }

    @Test
    public void testMAPE() {
        MAPE mape = new MAPE();
        double[] actual = {1.0f, 2.0f, 3.0f};
        double[] predicted = {1.0f, 2.0f, 3.0f};
        assertEquals(0.0f, mape.calculateError(actual, predicted), DELTA);
    }

}
