package sk.majba.backpropagationneuralnetwork.be;

import io.fair_acc.dataset.spi.DoubleDataSet;
import javafx.application.Platform;
import sk.majba.backpropagationneuralnetwork.be.error_metrics.ErrorMetric;
import sk.majba.backpropagationneuralnetwork.be.utils.DatasetUtils;
import sk.majba.backpropagationneuralnetwork.be.utils.MatrixUtils;

import java.io.IOException;
import java.util.*;

public class Network {
    private final List<Layer> layers = new ArrayList<>();
    private DoubleDataSet trainSeries;
    private DoubleDataSet testSeries;
    public static final Random random = new Random(1);
    private ErrorMetric errorMetric;
    private List<Layer> bestWeights;

    public Network(ErrorMetric errorMetric,DoubleDataSet trainSeries,DoubleDataSet testSeries) {
        this.errorMetric = errorMetric;
        this.trainSeries = trainSeries;
        this.testSeries = testSeries;
    }

    public void initNetwork() {
        for (int i = 1; i < this.layers.size(); i++) {
            Layer previousLayer = this.layers.get(i - 1);
            Layer currentLayer = this.layers.get(i);

            currentLayer.initWeights(random, previousLayer.getNeuronCount());
        }
    }

    public void addLayer(Layer layer) {
        this.layers.add(layer);
    }

    public void addHiddenLayer(Layer layer, int index) {
        this.layers.add(index, layer);
    }

    public void forwardPass(double[][] input) {
        // set the output of the input layer as the input data
        this.layers.getFirst().setInput(input);
        this.layers.getFirst().setOutput(input);

        for (int i = 1; i < this.layers.size(); i++) {
            Layer previousLayer = this.layers.get(i - 1);
            Layer currentLayer = this.layers.get(i);

            currentLayer.calculateInput(previousLayer.getOutput());
            currentLayer.calculateOutput();
        }
    }

    public double[][] backwardPass(double[][] targetOutput, double learningRate) {
        Layer outputLayer = this.layers.getLast();
        double[][] output = outputLayer.getOutput();
        double[][] predictionError = new double[output.length][output[0].length];
        double[][] outputLayerOutputDerivation = new double[output.length][output[0].length];

        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < output[0].length; j++) {
                outputLayerOutputDerivation[i][j] = outputLayer.getActivationFunction().derivative(output[i][j]);
                predictionError[i] = this.errorMetric.calculateErrorDerivative(targetOutput[i], output[i]);
            }
        }

        for (int i = 0; i < predictionError.length; i++) {
            for (int j = 0; j < predictionError[0].length; j++) {
                predictionError[i][j] *= -1;
            }
        }

        // -e * o(vystup)
        double[][] layerDeltas = MatrixUtils.multiply(predictionError, outputLayerOutputDerivation);

        // -e * o(vystup) * h1(vystup)
        outputLayer.setDelta(MatrixUtils.multiply(MatrixUtils.transpose(this.layers.get(this.layers.size() - 2).getOutput()), layerDeltas));

        for (int i = this.layers.size() - 2; i > 0; i--) {
            Layer currentLayer = this.layers.get(i);
            output = currentLayer.getOutput();
            double[][] currentLayerOutputDerivation = new double[output.length][output[0].length];

            for (int j = 0; j < currentLayerOutputDerivation.length; j++) {
                for (int k = 0; k < currentLayerOutputDerivation[0].length; k++) {
                    currentLayerOutputDerivation[j][k] = currentLayer.getActivationFunction().derivative(output[j][k]);
                }
            }

            // delta * wi
            layerDeltas = MatrixUtils.multiply(this.layers.get(i + 1).getWeights(), layerDeltas);
            currentLayerOutputDerivation = MatrixUtils.transpose(currentLayerOutputDerivation);

            // delta* wi * psi
            for (int j = 0; j < currentLayerOutputDerivation.length; j++) {
                for (int k = 0; k < currentLayerOutputDerivation[0].length; k++) {
                    layerDeltas[j][k] *= currentLayerOutputDerivation[j][k];
                }
            }

            currentLayer.setDelta(MatrixUtils.multiply(MatrixUtils.transpose(this.layers.get(i - 1).getInput()), MatrixUtils.transpose(layerDeltas)));
        }

        // update layers weights
        for (int i = 1; i < this.layers.size(); i++) {
            this.layers.get(i).updateWeights(learningRate);
        }

        // output the prediction error
        return predictionError;
    }

    public void train(String datasetFilePath, double trainTestSplitRatio, int numberOfEpochs, double learningRate) throws IOException {
        // 1. Open dataset
        double[][] dataset = DatasetUtils.readCSV(datasetFilePath, ",");

        this.layers.getFirst().setNeuronCount(dataset[0].length - 1);
//        this.initNetwork();

        // 2. Split dataset
        Map<String, double[][]> trainTestMap = DatasetUtils.splitData(dataset, trainTestSplitRatio, false,
                dataset[0].length - 1, random);

        double[][] trainInput = trainTestMap.get("trainInput");
        double[][] trainOutput = trainTestMap.get("trainOutput");

        double[][] testInput = trainTestMap.get("testInput");
        double[][] testOutput = trainTestMap.get("testOutput");

        double[] trainPrediction = new double[trainInput.length];
        double[] testPrediction = new double[testInput.length];

        double minError = Double.MAX_VALUE;

        for (int i = 0; i < numberOfEpochs; i++) {
            // 4. Train network
            for (int j = 0; j < trainInput.length; j++) {
                this.forwardPass(new double[][]{trainInput[j]});

                // TODO make it more dynamic, right now it is only good for regression problems
                // the prediction layer is always a 1D array
                trainPrediction[j] = this.layers.getLast().getOutput()[0][0];

                this.backwardPass(new double[][]{trainOutput[j]}, learningRate);
            }

            // 5. Test network
            for (int j = 0; j < testInput.length; j++) {
                this.forwardPass(new double[][]{testInput[j]});
                testPrediction[j] = this.layers.getLast().getOutput()[0][0];

//                if (i == numberOfEpochs - 1) {
//                    System.out.printf("Real: %f, prediction: %f%n", testOutput[j][0], testPrediction[j]);
//                }
            }

            // 6. Process results
            double trainError = this.errorMetric.calculateError(MatrixUtils.transpose(trainOutput)[0], trainPrediction);
            double testError = this.errorMetric.calculateError(MatrixUtils.transpose(testOutput)[0], testPrediction);

            System.out.println("Loss after epoch " + (i + 1) + ": " + trainError);

            // Save best weights
            if (trainError < minError) {
                this.bestWeights = this.layers;
            }

            // Output data to chart
            final int finalI = i + 1;
            Platform.runLater(() -> {
                this.trainSeries.add(finalI, trainError);
                this.testSeries.add(finalI, testError);
            });
        }
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public void setTrainSeries(DoubleDataSet trainSeries) {
        this.trainSeries = trainSeries;
    }

    public void setTestSeries(DoubleDataSet testSeries) {
        this.testSeries = testSeries;
    }

    public void setErrorMetric(ErrorMetric errorMetric) {
        this.errorMetric = errorMetric;
    }
}
