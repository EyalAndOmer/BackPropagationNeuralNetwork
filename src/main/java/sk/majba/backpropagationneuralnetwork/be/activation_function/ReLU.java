package sk.majba.backpropagationneuralnetwork.be.activation_function;

public class ReLU extends ActivationFunction {
    @Override
    public double activate(double outputs) {
        return Math.max(0, outputs);
    }

    @Override
    public double derivative(double outputs) {
        return outputs > 0 ? 1 : 0;
    }

    @Override
    public String getName() {
        return "ReLU";
    }
}
