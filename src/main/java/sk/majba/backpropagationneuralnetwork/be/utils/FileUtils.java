package sk.majba.backpropagationneuralnetwork.be.utils;

import sk.majba.backpropagationneuralnetwork.be.Layer;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FileUtils {
    private FileUtils() {

    }

    public static void saveWeightsToFile(List<Layer> layerList, String filename) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            for (int i = 1; i < layerList.size(); i++) {
                writer.write(layerWeightsToString(layerList.get(i)));
                writer.newLine(); // Add an empty line between layers
            }
        }
    }

    private static String layerWeightsToString(Layer layer) {
        double[][] array = layer.getWeights();
        StringBuilder sb = new StringBuilder();
        sb.append(layer.getLayerName()).append(",").append(layer.getActivationFunction().getName()).append(":\n");
        sb.append("[\n");
        for (double[] row : array) {
            sb.append("[ ");
            for (double num : row) {
                sb.append(String.format("%.20f ", num));
            }
            sb.append("]\n");
        }
        sb.append("]");
        return sb.toString();
    }

    public static Map<String, double[][]> readLayersFromFile(String filename) throws IOException {
        Map<String, double[][]> layerWeights = new HashMap<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String line;
            String layerName = "";
            List<double[]> weights = new ArrayList<>();
            while ((line = reader.readLine()) != null) {
                if (line.isEmpty()) {
                    // End of a layer, add to map
                    layerWeights.put(layerName, weights.toArray(new double[0][]));
                    // Reset for next layer
                    layerName = "";
                    weights.clear();
                } else if (layerName.isEmpty()) {
                    // First line of a layer is the layer name
                    layerName = line;
                } else {
                    // Parse weights from line and add to list
                    String[] nums = line.substring(1, line.length() - 1).split(" ");
                    double[] row = new double[nums.length];
                    for (int i = 0; i < nums.length; i++) {
                        row[i] = Double.parseDouble(nums[i]);
                    }
                    weights.add(row);
                }
            }
        }
        return layerWeights;
    }
}
