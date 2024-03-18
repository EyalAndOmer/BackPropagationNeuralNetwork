package sk.majba.backpropagationneuralnetwork.be.activation_function;

public class Sigmoid extends ActivationFunction {
    @Override
    public double activate(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    @Override
    public double derivative(double x) {
        return x * (1 - x);
    }

    @Override
    public String getName() {
        return "sigmoid";
    }
}
