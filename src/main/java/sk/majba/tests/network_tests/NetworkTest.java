package sk.majba.tests.network_tests;

import org.junit.Assert;
import org.junit.Test;
import sk.majba.backpropagationneuralnetwork.be.Layer;
import sk.majba.backpropagationneuralnetwork.be.LayerType;
import sk.majba.backpropagationneuralnetwork.be.Network;
import sk.majba.backpropagationneuralnetwork.be.activation_function.Linear;
import sk.majba.backpropagationneuralnetwork.be.activation_function.ReLU;
import sk.majba.backpropagationneuralnetwork.be.activation_function.Sigmoid;
import sk.majba.backpropagationneuralnetwork.be.error_metrics.MAPE;
import sk.majba.backpropagationneuralnetwork.be.error_metrics.MSE;
import sk.majba.backpropagationneuralnetwork.be.utils.DatasetUtils;
import sk.majba.backpropagationneuralnetwork.be.utils.FileUtils;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class NetworkTest {
    private static final double DELTA = 1e-3;

    @Test
    public void testForwardPass1() {
        Network network = new Network(new MAPE(), null, null);

        // Initialize your layers here
        network.addLayer(new Layer(1, LayerType.INPUT, new Linear(), "input"));
        network.addLayer(new Layer(3, LayerType.HIDDEN, new Sigmoid(), "hidden 1"));
        network.addLayer(new Layer(1, LayerType.OUTPUT, new Linear(), "output"));
        network.initNetwork();

        network.getLayers().get(1).setWeights(new double[][]{new double[]{0.461756f, -0.179838f, -0.58457f}});
        network.getLayers().get(2).setWeights(new double[][]{new double[]{-0.33456f}, new double[]{0.93550f}, new double[]{-0.98770f}});

        network.forwardPass(new double[][]{new double[]{0.8f}});

        double[][] expectedOutput = new double[][]{new double[]{-0.1441f}};
        double[][] networkOutput = network.getLayers().getLast().getOutput();

        for (int i = 0; i < networkOutput.length; i++) {
            Assert.assertArrayEquals(expectedOutput[i], networkOutput[i], DELTA);

        }
    }

    @Test
    public void testForwardPass2() {
        Network network = new Network(new MAPE(), null, null);

        // Initialize your layers here
        network.addLayer(new Layer(3, LayerType.INPUT, new Linear(), "input"));
        network.addLayer(new Layer(3, LayerType.HIDDEN, new ReLU(), "hidden 1"));
        network.addLayer(new Layer(2, LayerType.HIDDEN, new Sigmoid(), "output"));
        network.addLayer(new Layer(1, LayerType.OUTPUT, new Linear(), "output"));
        network.initNetwork();

        network.getLayers().get(1).setWeights(new double[][]{
                new double[]{0.1f, 0.2f, 0.3f},
                new double[]{0.3f, 0.2f, 0.7f},
                new double[]{0.4f, 0.3f, 0.9f}
        });
        network.getLayers().get(2).setWeights(new double[][]{
                new double[]{0.2f, 0.3f},
                new double[]{0.3f, 0.5f},
                new double[]{0.6f, 0.4f}
        });
        network.getLayers().get(3).setWeights(new double[][]{new double[]{0.1f}, new double[]{0.3f}});

        network.forwardPass(new double[][]{new double[]{0.1f, 0.2f, 0.7f}});

        double[][] expectedOutput = new double[][]{new double[]{0.256f}};
        double[][] networkOutput = network.getLayers().getLast().getOutput();

        for (int i = 0; i < expectedOutput.length; i++) {
            Assert.assertArrayEquals(networkOutput[i], expectedOutput[i], DELTA);
        }
    }

    @Test
    public void backpropagationTest1() {
        Network network = new Network(new MAPE(), null, null);

        // Initialize your layers here
        network.addLayer(new Layer(1, LayerType.INPUT, new Linear(), "input"));
        network.addLayer(new Layer(3, LayerType.HIDDEN, new Sigmoid(), "hidden 1"));
        network.addLayer(new Layer(1, LayerType.OUTPUT, new Linear(), "output"));
        network.initNetwork();

        network.getLayers().get(1).setWeights(new double[][]{new double[]{0.461756f, -0.179838f, -0.58457f}});
        network.getLayers().get(2).setWeights(new double[][]{new double[]{-0.33456f}, new double[]{0.93550f}, new double[]{-0.98770f}});

        network.forwardPass(new double[][]{new double[]{0.8f}});
        network.backwardPass(new double[][]{new double[]{0.717356091f}}, 0.01f);


        double[][] hiddenLayerUpdatedWeights = new double[][]{new double[]{0.46120729f, -0.178227668f, -0.586176559f}};
        double[][] outputLayerUpdatedWeights = new double[][]{new double[]{-0.32947f}, new double[]{0.93950f}, new double[]{-0.98438f}};

        Assert.assertArrayEquals(outputLayerUpdatedWeights, network.getLayers().getLast().getWeights());
        Assert.assertArrayEquals(hiddenLayerUpdatedWeights, network.getLayers().get(network.getLayers().size() - 2).getWeights());
    }

    @Test
    public void backpropagationTest2() throws IOException {
        Network network = new Network(new MSE(), null, null);

        // Initialize your layers here
        network.addLayer(new Layer(3, LayerType.INPUT, new Linear(), "input"));
        network.addLayer(new Layer(3, LayerType.HIDDEN, new ReLU(), "hidden 1"));
        network.addLayer(new Layer(2, LayerType.HIDDEN, new Sigmoid(), "output"));
        network.addLayer(new Layer(1, LayerType.OUTPUT, new Linear(), "output"));
        network.initNetwork();

        network.getLayers().get(1).setWeights(new double[][]{
                new double[]{0.1, 0.2, 0.3},
                new double[]{0.3, 0.2, 0.7},
                new double[]{0.4, 0.3, 0.9}
        });
        network.getLayers().get(2).setWeights(new double[][]{
                new double[]{0.2, 0.3},
                new double[]{0.3, 0.5},
                new double[]{0.6, 0.4}
        });
        network.getLayers().get(3).setWeights(new double[][]{new double[]{0.1}, new double[]{0.3}});

        network.forwardPass(new double[][]{new double[]{0.1, 0.2, 0.7}});
        network.backwardPass(new double[][]{new double[]{0.5}}, 0.01);

        double[][] hiddenLayer1UpdatedWeights = new double[][]{
                new double[]{0.10001237f, 0.20002026f, 0.30002019f},
                new double[]{0.30002475f, 0.20004051f, 0.70004038f},
                new double[]{0.40008663f, 0.30014180f, 0.90014132f}
        };

        double[][] hiddenLayer2UpdatedWeights = new double[][]{
                new double[]{0.2000387f, 0.3001185f},
                new double[]{0.3000298f, 0.500914f},
                new double[]{0.6000885f, 0.400271f}
        };

        double[][] outputLayerUpdatedWeights = new double[][]{new double[]{0.10318f}, new double[]{0.3031f}};

        for (int i = 0; i < outputLayerUpdatedWeights.length; i++) {
            Assert.assertArrayEquals(outputLayerUpdatedWeights[i], network.getLayers().getLast().getWeights()[i], DELTA);
        }

        for (int i = 0; i < hiddenLayer2UpdatedWeights.length; i++) {
            Assert.assertArrayEquals(hiddenLayer2UpdatedWeights[i], network.getLayers().get(network.getLayers().size() - 2).getWeights()[i], DELTA);
        }

        for (int i = 0; i < hiddenLayer1UpdatedWeights.length; i++) {
            Assert.assertArrayEquals(hiddenLayer1UpdatedWeights[i], network.getLayers().get(network.getLayers().size() - 3).getWeights()[i], DELTA);
        }

//        List<HashMap<String, Object>> jsonData = DatasetUtils.readJsonFile("C:\\personal\\ING\\1. Semester\\projekt 1\\BackPropagationNeuralNetwork\\datasets\\python_outputs.txt");
//
//        Arrays.deepEquals(((double[][][]) jsonData.get(0).get("Weights"))[0], network.getLayers().get(1).getWeights());
//        System.out.println();
    }

    @Test
    public void backPropagationTest3() throws IOException {
        List<HashMap<String, Object>> jsonData = DatasetUtils.readJsonFile("C:\\personal\\ING\\1. Semester\\projekt 1\\BackPropagationNeuralNetwork\\datasets\\python_outputs.txt");
        System.out.println(jsonData);
    }

    @Test
    public void sineDatasetTest() throws IOException {
        Network network = new Network(new MSE(), null, null);

        // Initialize your layers here
        network.addLayer(new Layer(1, LayerType.INPUT, new Linear(), "input"));
        network.addLayer(new Layer(64, LayerType.HIDDEN, new ReLU(), "hidden1"));
        network.addLayer(new Layer(32, LayerType.HIDDEN, new ReLU(), "hidden2"));
        network.addLayer(new Layer(8, LayerType.HIDDEN, new Sigmoid(), "hidden3"));
        network.addLayer(new Layer(1, LayerType.OUTPUT, new Linear(), "output"));
        network.initNetwork();

        FileUtils.saveWeightsToFile(network.getLayers(), "sine_layers.txt");

        network.train("C:\\personal\\ING\\1. Semester\\projekt 1\\BackPropagationNeuralNetwork\\datasets\\sine_values.csv",
                0.8, 20, 0.0001);
//        System.out.println(network.getLayers().getLast().getOutput()[0][0]);
    }
}
