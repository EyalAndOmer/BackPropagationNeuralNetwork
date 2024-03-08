package sk.majba.backpropagationneuralnetwork.be.activation_function;

public abstract class ActivationFunction {
    public abstract double activate(double outputs);
    public abstract double derivative(double outputs);
    public abstract String getName();

}
