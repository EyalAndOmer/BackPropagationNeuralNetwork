package sk.majba.backpropagationneuralnetwork.be.error_metrics;

public class MAPE extends ErrorMetric {
    @Override
    public double calculateError(double[] actual, double[] predicted) {
        // Implement Mean Absolute Percentage Error calculation
        double sum = 0;
        for (int i = 0; i < actual.length; i++) {
            sum += Math.abs((actual[i] - predicted[i]) / actual[i]);
        }
        return (sum / actual.length) * 100;
    }

    @Override
    public double[] calculateErrorDerivative(double[] actual, double[] predicted) {
        int n = actual.length;
        double[] mapeDerivative = new double[n];
        for (int i = 0; i < n; i++) {
            double mape = Math.abs((actual[i] - predicted[i]) / actual[i]);
            mapeDerivative[i] = (100 / mape) * ((actual[i] / Math.pow(predicted[i], 2)) * ((predicted[i] - actual[i]) / Math.abs(predicted[i] - actual[i])));
        }
        return mapeDerivative;
    }
}
