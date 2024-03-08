package sk.majba.backpropagationneuralnetwork.be;


import sk.majba.backpropagationneuralnetwork.be.activation_function.ActivationFunction;
import sk.majba.backpropagationneuralnetwork.be.utils.MatrixUtils;

import java.util.Random;

public class Layer {
    private String layerName;
    private int neuronCount;
    private LayerType layerType;
    private ActivationFunction activationFunction;
    private double[][] weights;
    private double[][] input;
    private double[][] output;
    private double[][] delta;

    public Layer(int neuronCount, LayerType layerType, ActivationFunction activationFunction, String layerName) {
        this.neuronCount = neuronCount;
        this.layerType = layerType;
        this.activationFunction = activationFunction;
        this.layerName = layerName;
    }

    // Random weight initialization from -1 to 1
    public void initWeights(Random random, int previousLayerNeuronCount) {
        this.weights = new double[previousLayerNeuronCount][this.neuronCount];
        for (int i = 0; i  < this.weights.length; i++) {
            for (int j = 0; j < this.weights[i].length; j++) {
                this.weights[i][j] = random.nextDouble() * 2 - 1;
            }
        }
    }

    // Random weight initialization Xavier
//    public void initWeights(Random random, int previousLayerNeuronCount) {
//        this.weights = new double[previousLayerNeuronCount][this.neuronCount];
//        double limit = Math.sqrt(6.0 / (previousLayerNeuronCount + this.neuronCount));
//        for (int i = 0; i < this.weights.length; i++) {
//            for (int j = 0; j < this.weights[i].length; j++) {
//                this.weights[i][j] = random.nextDouble() * 2 * limit - limit;
//            }
//        }
//    }

    public double[][] getWeights() {
        return weights;
    }

    public void calculateOutput() {
        double[][] activatedInput = new double[this.input.length][this.input[0].length];
        for (int i = 0; i < this.input.length; i++) {
            for (int j = 0; j < this.input[0].length; j++) {
                activatedInput[i][j] =  this.activationFunction.activate(this.input[i][j]);
            }
        }
        this.output = activatedInput;
    }

    public void calculateInput(double[][] previousLayerOutput) {
        this.input = MatrixUtils.multiply(previousLayerOutput, this.weights);
    }

    public void setWeights(double[][] weights) {
        this.weights = weights;
    }

    public int getNeuronCount() {
        return neuronCount;
    }

    public LayerType getLayerType() {
        return layerType;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public void setNeuronCount(int neuronCount) {
        this.neuronCount = neuronCount;
    }

    public void setLayerType(LayerType layerType) {
        this.layerType = layerType;
    }

    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    public String getLayerName() {
        return layerName;
    }

    public void setLayerName(String layerName) {
        this.layerName = layerName;
    }

    public double[][] getInput() {
        return input;
    }

    public void setInput(double[][] input) {
        this.input = input;
    }

    public double[][] getOutput() {
        return output;
    }

    public void setOutput(double[][] output) {
        this.output = output;
    }

    public void setDelta(double[][] delta) {
        this.delta = delta;
    }

    public void updateWeights(double learningRate) {
        for (int i = 0; i < this.weights.length; i++) {
            for (int j = 0; j < this.weights[0].length; j++) {
                this.weights[i][j] -= learningRate * this.delta[i][j];
            }
        }
    }
}
