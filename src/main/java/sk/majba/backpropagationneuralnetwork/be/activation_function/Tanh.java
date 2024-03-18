package sk.majba.backpropagationneuralnetwork.be.activation_function;

public class Tanh extends ActivationFunction{
    @Override
    public double activate(double outputs) {
        return Math.tanh(outputs);
    }

    @Override
    public double derivative(double outputs) {
        double tanhX = Math.tanh(outputs);
        return 1 - tanhX * tanhX;
    }

    @Override
    public String getName() {
        return "tanh";
    }
}
