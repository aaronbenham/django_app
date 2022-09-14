function update_deviation_graph(data) {
    StandardDeviationGraph.config.data.datasets[0].data.push(data);
    StandardDeviationGraph.config.data.datasets[0].data.shift();
    StandardDeviationGraph.update();
};

function update_confidence_graph(overall_confidence, above_threshold) {
    AverageConfidenceGraph.config.data.datasets[1].data.push(above_threshold);
    AverageConfidenceGraph.config.data.datasets[1].data.shift();

    AverageConfidenceGraph.config.data.datasets[0].data.push(overall_confidence);
    AverageConfidenceGraph.config.data.datasets[0].data.shift();

    AverageConfidenceGraph.update();
};

function update_bounding_box_graph(data) {
    NumberOfBoundingBoxes.config.data.datasets[0].data.push(data);
    NumberOfBoundingBoxes.config.data.datasets[0].data.shift();
    NumberOfBoundingBoxes.update();
};