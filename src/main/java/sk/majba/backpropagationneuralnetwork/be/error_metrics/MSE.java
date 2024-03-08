package sk.majba.backpropagationneuralnetwork.be.error_metrics;

public class MSE extends ErrorMetric {
    @Override
    public double calculateError(double[] actual, double[] predicted) {
        double sum = 0;
        for (int i = 0; i < actual.length; i++) {
            sum += Math.pow(actual[i] - predicted[i], 2);
        }
        return sum / actual.length;
    }

    @Override
    public double[] calculateErrorDerivative(double[] actual, double[] predicted) {
        int n = actual.length;
        double[] mseDerivative = new double[n];
        for (int i = 0; i < n; i++) {
            mseDerivative[i] = 2 * (actual[i] - predicted[0]);
        }
        return mseDerivative;
    }
}
