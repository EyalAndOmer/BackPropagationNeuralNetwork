module sk.majba.backpropagationneuralnetwork.fe {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.controlsfx.controls;
    requires net.synedra.validatorfx;
    requires org.kordamp.ikonli.javafx;
    requires org.kordamp.bootstrapfx.core;
    requires junit;
    requires org.json;
    requires io.fair_acc.chartfx;
    requires io.fair_acc.dataset;


    exports sk.majba.backpropagationneuralnetwork.fe;
    opens sk.majba.backpropagationneuralnetwork.fe to javafx.fxml;

    exports sk.majba.tests.matrix_tests;
    opens sk.majba.tests.matrix_tests to org.junit.jupiter.api;

    exports sk.majba.tests.network_tests;
    opens sk.majba.tests.network_tests to org.junit.jupiter.api;
}
