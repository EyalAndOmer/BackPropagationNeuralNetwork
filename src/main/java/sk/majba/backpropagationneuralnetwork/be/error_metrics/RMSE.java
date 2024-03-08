package sk.majba.backpropagationneuralnetwork.be.error_metrics;

public class RMSE extends ErrorMetric {
    @Override
    public double calculateError(double[] actual, double[] predicted) {
        // Implement Root Mean Squared Error calculation
        double sum = 0;
        for (int i = 0; i < actual.length; i++) {
            sum += Math.pow(actual[i] - predicted[i], 2);
        }
        return Math.sqrt(sum / actual.length);
    }

    @Override
    public double[] calculateErrorDerivative(double[] actual, double[] predicted) {
        int n = actual.length;
        double[] rmseDerivative = new double[n];
        for (int i = 0; i < n; i++) {
            double rmse = Math.sqrt(Math.pow((actual[i] - predicted[i]), 2));
            rmseDerivative[i] = (actual[i] - predicted[i]) / rmse;
        }
        return rmseDerivative;
    }
}
