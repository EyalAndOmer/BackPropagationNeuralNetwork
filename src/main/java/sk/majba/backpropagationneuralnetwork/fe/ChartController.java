package sk.majba.backpropagationneuralnetwork.fe;

import io.fair_acc.chartfx.plugins.Zoomer;
import io.fair_acc.chartfx.renderer.ErrorStyle;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.StackPane;
import javafx.stage.FileChooser;
import javafx.stage.Modality;
import javafx.stage.Stage;
import sk.majba.backpropagationneuralnetwork.be.Layer;
import sk.majba.backpropagationneuralnetwork.be.LayerType;
import sk.majba.backpropagationneuralnetwork.be.Network;
import sk.majba.backpropagationneuralnetwork.be.activation_function.*;
import sk.majba.backpropagationneuralnetwork.be.error_metrics.ErrorMetric;
import sk.majba.backpropagationneuralnetwork.be.error_metrics.MSE;
import sk.majba.backpropagationneuralnetwork.be.error_metrics.RMSE;
import io.fair_acc.dataset.spi.DoubleDataSet;

import io.fair_acc.chartfx.renderer.spi.ErrorDataSetRenderer;


import java.io.File;
import java.io.IOException;
import java.security.InvalidParameterException;

public class ChartController {
    @FXML
    public StackPane chartContainer;
    @FXML
    public io.fair_acc.chartfx.XYChart lineChart;
    @FXML
    public Button btnStartTraining;
    @FXML
    public TextField txtFieldNumberOfEpochs;
    @FXML
    public ChoiceBox<String> choiceErrorFunction;
    @FXML
    public TextField txtFieldLearningRate;
    @FXML
    public TextField txtFieldTrainTestSplit;
    public Label labelSelectedDatasetFile;
    @FXML
    private Button btnVrstvy;

    @FXML
    private ListView<HBox> layersListView;

    private FileChooser datasetFileChooser;
    private Network network;
    private String datasetPath;
    private File lastVisitedDirectory;

    public void initialize() {
        txtFieldNumberOfEpochs.setText("1000");
        choiceErrorFunction.setValue("MSE");
        txtFieldLearningRate.setText("0.001");
        txtFieldTrainTestSplit.setText("0.8");

        // Zoom
        lineChart.getPlugins().add(new Zoomer());

        // Make the lines thinner
        lineChart.setStyle("-fx-stroke-width: 0.5px;");

        // Generate datasets to the chart
        final DoubleDataSet dataSet1 = new DoubleDataSet("Train dataset");
        final DoubleDataSet dataSet2 = new DoubleDataSet("Test dataset");

        var avgRenderer = new ErrorDataSetRenderer();
        avgRenderer.setDrawMarker(false);
        avgRenderer.setErrorStyle(ErrorStyle.NONE);
        avgRenderer.getDatasets().addAll(dataSet1, dataSet2);
        lineChart.getRenderers().add(avgRenderer);


        // Network init
        this.network = new Network(new MSE(), dataSet1, dataSet2);
        this.network.addLayer(new Layer(1, LayerType.INPUT, new Linear(), "input"));
        this.network.addLayer(new Layer(32, LayerType.HIDDEN, new ReLU(), "hidden 3"));
        this.network.addLayer(new Layer(16, LayerType.HIDDEN, new ReLU(), "hidden 3"));
        this.network.addLayer(new Layer(8, LayerType.HIDDEN, new Sigmoid(), "hidden 3"));
        this.network.addLayer(new Layer(1, LayerType.OUTPUT, new Linear(), "output"));
        this.network.initNetwork();

        this.updateLayersListView();

        // Set button action
        btnVrstvy.setOnAction(this::handleButtonAction);

        // set choiceBox items
        ObservableList<String> items = FXCollections.observableArrayList("MSE", "RMSE");
        this.choiceErrorFunction.setItems(items);

        // File chooser
        this.datasetFileChooser = new FileChooser();
        this.datasetFileChooser.setInitialDirectory(new File("C:\\personal\\ING\\1. Semester\\projekt 1\\BackPropagationNeuralNetwork\\datasets"));
    }

    public void handleButtonAction(ActionEvent event) {
        this.openAddLayerModal(null);
    }

    private void openAddLayerModal(Layer layerToEdit) {
        Stage modal = new Stage();
        modal.initModality(Modality.APPLICATION_MODAL);

        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(10);
        grid.setPadding(new Insets(20, 20, 20, 20));

        TextField layerNameField = new TextField(layerToEdit != null ? String.valueOf(layerToEdit.getLayerName()) : "");
        TextField neuronCountField = new TextField(layerToEdit != null ? String.valueOf(layerToEdit.getNeuronCount()) : "");
        ChoiceBox<String> activationFunctionChoiceBox = new ChoiceBox<>(FXCollections.observableArrayList("Sigmoid", "ReLU", "Linear", "Tanh"));
        activationFunctionChoiceBox.setValue(layerToEdit != null ? layerToEdit.getActivationFunction().getName() : "Sigmoid");
        ChoiceBox<String> layerTypeChoiceBox = new ChoiceBox<>(FXCollections.observableArrayList("HIDDEN"));
        layerTypeChoiceBox.setValue(layerToEdit != null ? layerToEdit.getLayerType().toString() : "HIDDEN");
        layerTypeChoiceBox.setDisable(true);

        Button btnAdd = new Button(layerToEdit != null ? "Uprav" : "Pridaj");
        btnAdd.setOnAction(e -> {
            int neuronCount = Integer.parseInt(neuronCountField.getText());
            ActivationFunction activationFunction = this.selectActivationFunction(activationFunctionChoiceBox.getValue());
            String layerType = layerTypeChoiceBox.getValue();
            String layerName = layerNameField.getText();

            if (layerToEdit != null) {
                layerToEdit.setLayerName(layerName);
                layerToEdit.setNeuronCount(neuronCount);
                layerToEdit.setActivationFunction(activationFunction);
                layerToEdit.setLayerType(LayerType.valueOf(layerType));
            } else {
                this.network.addHiddenLayer(new Layer(neuronCount, LayerType.valueOf(layerType), activationFunction, layerName), this.network.getLayers().size() - 1);
            }

            this.network.initNetwork();
            updateLayersListView();

            modal.close();
        });

        Button btnCancel = new Button("Zrušiť");
        btnCancel.setOnAction(e -> modal.close());

        grid.add(new Label("Názov vrstvy:"), 0, 0);
        grid.add(layerNameField, 1, 0);
        grid.add(new Label("Počet neurónov:"), 0, 1);
        grid.add(neuronCountField, 1, 1);
        grid.add(new Label("Aktivačná vrstva:"), 0, 2);
        grid.add(activationFunctionChoiceBox, 1, 2);
        grid.add(new Label("Typ vrstvy:"), 0, 3);
        grid.add(layerTypeChoiceBox, 1, 3);
        grid.add(btnAdd, 0, 4);
        grid.add(btnCancel, 1, 4);

        Scene modalScene = new Scene(grid, 300, 200);
        modal.setScene(modalScene);
        modal.showAndWait();
    }

    private void updateLayersListView() {
        layersListView.getItems().clear();
        for (int i = 0; i < this.network.getLayers().size(); i++) {
            Layer layer = this.network.getLayers().get(i);
            HBox layerBox = new HBox(10); // Add spacing between elements
            layerBox.setPadding(new Insets(5, 5, 5, 5)); // Add padding around the HBox
            layerBox.setAlignment(Pos.CENTER_LEFT); // Align the content to the left

            Label layerInfo = new Label("Názov vrstvy: " + layer.getLayerName() + ", Typ vrstvy: " + layer.getLayerType() + ", Počet neurónov: " + layer.getNeuronCount() + ", Aktivačná funkcia: " + layer.getActivationFunction().getName());
            layerInfo.setStyle("-fx-font-weight: bold"); // Make the layer info bold

            Button editButton = new Button("Upraviť");
            editButton.setStyle("-fx-background-color: #90ee90"); // Make the edit button green
            editButton.setOnAction(e -> openAddLayerModal(layer));

            Button deleteButton = new Button("Zmazať");
            deleteButton.setStyle("-fx-background-color: #ff6347"); // Make the delete button red
            deleteButton.setOnAction(e -> {
                this.network.getLayers().remove(layer);
                updateLayersListView();
            });

            if (i != 0 && i != this.network.getLayers().size() - 1) {
                Button upButton = new Button("Hore");
                upButton.setOnAction(e -> {
                    int index = this.network.getLayers().indexOf(layer);
                    if (index > 1) {
                        this.network.getLayers().remove(layer);
                        this.network.getLayers().add(index - 1, layer);
                        updateLayersListView();
                    }
                });

                Button downButton = new Button("Dole");
                downButton.setOnAction(e -> {
                    int index = this.network.getLayers().indexOf(layer);
                    if (index < this.network.getLayers().size() - 2) {
                        this.network.getLayers().remove(layer);
                        this.network.getLayers().add(index + 1, layer);
                        updateLayersListView();
                    }
                });

                layerBox.getChildren().addAll(layerInfo, editButton, deleteButton, upButton, downButton);
            } else {
                layerBox.getChildren().addAll(layerInfo, editButton);
                if (i != 0 && i != this.network.getLayers().size() - 1) {
                    layerBox.getChildren().add(deleteButton);
                }
            }

            layersListView.getItems().add(layerBox);
        }
    }

    private ActivationFunction selectActivationFunction(String activationFunctionName) {
        return switch (activationFunctionName) {
            case "Linear" -> new Linear();
            case "Sigmoid" -> new Sigmoid();
            case "ReLU" -> new ReLU();
            case "Tanh" -> new Tanh();
            default -> throw new InvalidParameterException("Activation function with such a name does not exist.");
        };
    }

    private ErrorMetric selectErrorMetric(String errorMetricName) {
        return switch (errorMetricName) {
            case "MSE" -> new MSE();
            case "RMSE" -> new RMSE();
            default -> throw new InvalidParameterException("Activation function with such a name does not exist.");
        };
    }

    public void launchNetworkTraining() {
        this.network.initNetwork();
        this.btnStartTraining.setDisable(true);
        this.network.setErrorMetric(this.selectErrorMetric(choiceErrorFunction.getValue()));

        if (this.lineChart.getRenderers().get(1).getDatasets().get(0).getDataCount() > 0) {
            this.lineChart.getRenderers().remove(1);

            final DoubleDataSet dataSet1 = new DoubleDataSet("Train dataset");
            final DoubleDataSet dataSet2 = new DoubleDataSet("Test dataset");

            this.network.setTrainSeries(dataSet1);
            this.network.setTestSeries(dataSet2);

            var avgRenderer = new ErrorDataSetRenderer();
            avgRenderer.setDrawMarker(false);
            avgRenderer.setErrorStyle(ErrorStyle.NONE);
            avgRenderer.getDatasets().addAll(dataSet1, dataSet2);
            this.lineChart.getRenderers().add(avgRenderer);
        }

        new Thread(() -> {
            try {
                this.network.train(this.datasetPath,
                        Double.parseDouble(this.txtFieldTrainTestSplit.getText()), Integer.parseInt(this.txtFieldNumberOfEpochs.getText()),
                        Double.parseDouble(this.txtFieldLearningRate.getText()));
            } catch (IOException e) {
                throw new RuntimeException(e);
            } finally {
                this.btnStartTraining.setDisable(false);
            }
        }).start();
    }

    public void selectDataset() {
        Stage currentStage = (Stage)btnStartTraining.getScene().getWindow();
        File selectedFile = this.datasetFileChooser.showOpenDialog(currentStage);

        if (this.lastVisitedDirectory != null && this.lastVisitedDirectory.isDirectory()) {
            datasetFileChooser.setInitialDirectory(this.lastVisitedDirectory);
        }

        if (selectedFile != null) {
            this.datasetPath = selectedFile.getAbsolutePath();
            this.labelSelectedDatasetFile.setText(selectedFile.getName());
            this.lastVisitedDirectory = selectedFile.getParentFile();
        }
    }
}
