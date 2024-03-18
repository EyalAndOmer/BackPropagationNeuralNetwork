package sk.majba.backpropagationneuralnetwork.be.activation_function;

public class Linear extends ActivationFunction {
    @Override
    public double activate(double outputs) {
        return outputs;
    }

    @Override
    public double derivative(double outputs) {
        return 1;
    }

    @Override
    public String getName() {
        return "linear";
    }
}
