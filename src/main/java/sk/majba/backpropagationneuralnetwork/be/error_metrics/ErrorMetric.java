package sk.majba.backpropagationneuralnetwork.be.error_metrics;

public abstract class ErrorMetric {
    public abstract double calculateError(double[] actual, double[] predicted);
    public abstract double[] calculateErrorDerivative(double[] actual, double[] predicted);
}
