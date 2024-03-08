package sk.majba.backpropagationneuralnetwork.be.utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

import org.json.JSONArray;
import org.json.JSONObject;

public class DatasetUtils {
    private DatasetUtils() {}
    public static Map<String, double[][]> splitData(double[][] dataset, double splitRatio, boolean shuffle, int inputSize, Random random) {
        // Calculate the size of the training set
        int trainSize = (int) Math.round(dataset.length * splitRatio);

        // Initialize the training and testing sets
        double[][] trainSet = new double[trainSize][];
        double[][] testSet = Arrays.copyOf(dataset, dataset.length);

        // Shuffle the test set if required
        if (shuffle) {
            for (int i = 0; i < testSet.length; i++) {
                int randomIndexToSwap = random.nextInt(testSet.length);
                double[] temp = testSet[randomIndexToSwap];
                testSet[randomIndexToSwap] = testSet[i];
                testSet[i] = temp;
            }
        }

        // Copy the first part of the shuffled array to the training set
        System.arraycopy(testSet, 0, trainSet, 0, trainSize);

        // Create a new test set from the remaining part of the shuffled array
        double[][] newTestSet = new double[testSet.length - trainSize][];
        System.arraycopy(testSet, trainSize, newTestSet, 0, newTestSet.length);

        // Split the training and testing data into inputs and outputs
        double[][] trainInput = new double[trainSize][inputSize];
        double[][] trainOutput = new double[trainSize][];
        double[][] testInput = new double[newTestSet.length][inputSize];
        double[][] testOutput = new double[newTestSet.length][];

        for (int i = 0; i < trainSize; i++) {
            System.arraycopy(trainSet[i], 0, trainInput[i], 0, inputSize);
            trainOutput[i] = Arrays.copyOfRange(trainSet[i], inputSize, trainSet[i].length);
        }

        for (int i = 0; i < newTestSet.length; i++) {
            System.arraycopy(newTestSet[i], 0, testInput[i], 0, inputSize);
            testOutput[i] = Arrays.copyOfRange(newTestSet[i], inputSize, newTestSet[i].length);
        }

        // Create a map to store the split data
        Map<String, double[][]> splitData = new HashMap<>();
        splitData.put("trainInput", trainInput);
        splitData.put("trainOutput", trainOutput);
        splitData.put("testInput", testInput);
        splitData.put("testOutput", testOutput);

        return splitData;
    }


    public static double[][] readCSV(String filePath, String delimiter) throws IOException, NumberFormatException {
        List<double[]> data = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(delimiter);
                double[] floatValues = new double[values.length];
                for (int i = 0; i < values.length; i++) {
                    try {
                        floatValues[i] = Double.parseDouble(values[i]);
                    } catch (NumberFormatException e) {
                        throw new NumberFormatException("Value is not numeric: " + values[i]);
                    }
                }
                data.add(floatValues);
            }
        }

        double[][] dataArray = new double[data.size()][];
        for (int i = 0; i < data.size(); i++) {
            dataArray[i] = data.get(i);
        }

        return dataArray;
    }

    public static List<HashMap<String, Object>> readJsonFile(String filePath) throws IOException {
        List<HashMap<String, Object>> data = new ArrayList<>();
        List<String> lines = Files.readAllLines(Paths.get(filePath));

        for (String line : lines) {
            JSONObject jsonObject = new JSONObject(line);
            HashMap<String, Object> map = new HashMap<>();

            // Parse Output
            JSONArray outputJsonArray = jsonObject.getJSONArray("Output");
            double[][] output = new double[outputJsonArray.length()][];
            for (int i = 0; i < outputJsonArray.length(); i++) {
                JSONArray row = outputJsonArray.getJSONArray(i);
                output[i] = new double[row.length()];
                for (int j = 0; j < row.length(); j++) {
                    output[i][j] = row.getDouble(j);
                }
            }
            map.put("Output", output);

            // Parse Weights
            JSONArray weightsJsonArray = jsonObject.getJSONArray("Weights");
            double[][][] weights = new double[weightsJsonArray.length()][][];
            for (int i = 0; i < weightsJsonArray.length(); i++) {
                JSONArray weight2D = weightsJsonArray.getJSONArray(i);
                weights[i] = new double[weight2D.length()][];
                for (int j = 0; j < weight2D.length(); j++) {
                    JSONArray row = weight2D.getJSONArray(j);
                    weights[i][j] = new double[row.length()];
                    for (int k = 0; k < row.length(); k++) {
                        weights[i][j][k] = row.getDouble(k);
                    }
                }
            }
            map.put("Weights", weights);

            // Parse Loss
            double loss = jsonObject.getDouble("Loss");
            map.put("Loss", loss);

            data.add(map);
        }

        return data;
    }
}
