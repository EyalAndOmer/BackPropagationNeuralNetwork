package sk.majba.tests.matrix_tests;

import org.junit.Test;
import sk.majba.backpropagationneuralnetwork.be.activation_function.Linear;
import sk.majba.backpropagationneuralnetwork.be.activation_function.ReLU;
import sk.majba.backpropagationneuralnetwork.be.activation_function.Sigmoid;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class ActivationFunctionTest {
    private static final double DELTA = 1e-7f;

    @Test
    public void testLinear() {
        Linear linear = new Linear();
        double[] outputs = {0.0f, -1.0f, 1.0f, 2.0f};
        double[] expectedActivate = {0.0f, -1.0f, 1.0f, 2.0f};
        double[] expectedDerivative = {1.0f, 1.0f, 1.0f, 1.0f};

        for (int i = 0; i < expectedActivate.length - 1; i++) {
            assertEquals(expectedActivate[i], linear.activate(outputs[i]), DELTA);
        }

        for (int i = 0; i < expectedDerivative.length - 1; i++) {
            assertEquals(expectedDerivative[i], linear.derivative(outputs[i]), DELTA);
        }
    }

    @Test
    public void testSigmoid() {
        Sigmoid sigmoid = new Sigmoid();
        double[] outputs = {0.0f, -1.0f, 1.0f, 2.0f};
        double[] expectedActivate = {0.5f, 0.26894143f, 0.7310586f, 0.880797f};
        double[] expectedDerivative = {0.25f, 0.19661194f, 0.19661193f, 0.10499363f};
        for (int i = 0; i < expectedActivate.length - 1; i++) {
            assertEquals(expectedActivate[i], sigmoid.activate(outputs[i]), DELTA);
        }

        for (int i = 0; i < expectedDerivative.length - 1; i++) {
            assertEquals(expectedDerivative[i], sigmoid.derivative(outputs[i]), DELTA);
        }
    }

    @Test
    public void testReLU() {
        ReLU relu = new ReLU();
        double[] outputs = {1.0f, 2.0f, -3.0f, -4.0f};
        double[] expectedActivate = {1.0f, 2.0f, 0.0f, 0.0f};
        double[] expectedDerivative = {1.0f, 1.0f, 0.0f, 0.0f};
        for (int i = 0; i < expectedActivate.length - 1; i++) {
            assertEquals(expectedActivate[i], relu.activate(outputs[i]), DELTA);
        }

        for (int i = 0; i < expectedDerivative.length - 1; i++) {
            assertEquals(expectedDerivative[i], relu.derivative(outputs[i]), DELTA);
        }
    }

    @Test
    public void testLinear2() {
        Linear linear = new Linear();
        double[] outputs = {1.0f, 2.0f, 3.0f, 4.0f};
        double[] expectedActivate = {1.0f, 2.0f, 3.0f, 4.0f};
        double[] expectedDerivative = {1.0f, 1.0f, 1.0f, 1.0f};
        for (int i = 0; i < expectedActivate.length - 1; i++) {
            assertEquals(expectedActivate[i], linear.activate(outputs[i]), DELTA);
        }

        for (int i = 0; i < expectedDerivative.length - 1; i++) {
            assertEquals(expectedDerivative[i], linear.derivative(outputs[i]), DELTA);
        }
    }

    @Test
    public void testSigmoidLargeValues() {
        Sigmoid sigmoid = new Sigmoid();
        double[] outputs = {1000.0f, -1000.0f, 10000.0f, -10000.0f};
        double[] expectedActivate = {1.0f, 0.0f, 1.0f, 0.0f};
        double[] expectedDerivative = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = 0; i < expectedActivate.length - 1; i++) {
            assertEquals(expectedActivate[i], sigmoid.activate(outputs[i]), DELTA);
        }

        for (int i = 0; i < expectedDerivative.length - 1; i++) {
            assertEquals(expectedDerivative[i], sigmoid.derivative(outputs[i]), DELTA);
        }
    }

    @Test
    public void testSigmoidComplex() {
        Sigmoid sigmoid = new Sigmoid();
        double[] outputs = {0.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f, 5.0f, -5.0f};
        double[] expectedActivate = {0.5f, 0.2689414f, 0.8807971f, 0.1192029f, 0.95257413f, 0.04742587f, 0.98201376f, 0.01798621f, 0.99330715f, 0.00669285f};
        double[] expectedDerivative = {0.25f, 0.19661194f, 0.10499357f, 0.10499357f, 0.04517666f, 0.04517666f, 0.017662706f, 0.017662706f, 0.00664806f, 0.00664806f};
        for (int i = 0; i < expectedActivate.length - 1; i++) {
            assertEquals(expectedActivate[i], sigmoid.activate(outputs[i]), DELTA);
        }

        for (int i = 0; i < expectedDerivative.length - 1; i++) {
            assertEquals(expectedDerivative[i], sigmoid.derivative(outputs[i]), DELTA);
        }
    }

}
