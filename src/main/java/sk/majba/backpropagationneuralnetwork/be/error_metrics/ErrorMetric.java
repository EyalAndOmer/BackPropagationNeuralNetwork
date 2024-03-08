package sk.majba.backpropagationneuralnetwork.be.error_metrics;

import java.util.List;

public abstract class ErrorMetric {
    public abstract double calculateError(double[] actual, double[] predicted);
    public abstract double[] calculateErrorDerivative(double[] actual, double[] predicted);
}
