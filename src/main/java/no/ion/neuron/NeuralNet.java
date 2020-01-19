package no.ion.neuron;

import no.ion.neuron.tensor.Matrix;
import no.ion.neuron.tensor.Vector;
import no.ion.neuron.internal.Layer;
import no.ion.neuron.optimizer.ComputationId;
import no.ion.neuron.optimizer.LayerId;
import no.ion.neuron.optimizer.Optimizer;
import no.ion.neuron.transform.BiasTransform;
import no.ion.neuron.transform.IdentityTransform;
import no.ion.neuron.transform.MapTransform;
import no.ion.neuron.transform.mapper.Mapper;
import no.ion.neuron.transform.Transform;
import no.ion.neuron.transform.WeightTransform;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class NeuralNet {
    private final Optimizer optimizer;
    private final Map<LayerId, Layer> layerMap = new HashMap<>();
    private final List<Layer> layers = new ArrayList<>();

    public NeuralNet(int inputSize, Optimizer optimizer) {
        this.optimizer = optimizer;
        addTransform(new IdentityTransform(inputSize));
    }

    public Layer addLayer(Matrix weight, Vector bias, Mapper mapper) {
        if (weight.columns() != outputLayer().outputSize()) {
            throw new IllegalArgumentException("Last layer has size " + outputLayer().outputSize() +
                    " but weight has input size " + weight.columns());
        } else if (bias.size() != weight.rows()) {
            throw new IllegalArgumentException("Weight output size " + weight.rows() +
                    " does not match bias size " + bias.size());
        }

        addTransform(new WeightTransform(weight));
        addTransform(new BiasTransform(bias));
        addTransform(new MapTransform(bias.size(), mapper));

        return outputLayer();
    }

    public Layer addTransform(Transform transform) {
        Layer layer = new Layer(transform, optimizer);

        if (outputLayer() != null) {
            layer.setUpstreamLayer(outputLayer());
            outputLayer().setDownstreamLayer(layer);
        }

        layerMap.put(layer.layerId(), layer);
        layers.add(layer);
        return layer;
    }

    public Vector compute(Vector input, Vector idealOutput) {
        var computationId = ComputationId.createNext();
        optimizer.startComputation(computationId, input, idealOutput);
        Layer.Result result = inputLayer().process(computationId, input, idealOutput);
        optimizer.endComputation(computationId);
        return result.netOutput();
    }

    public int inputSize() { return inputLayer().inputSize(); }
    public int outputSize() { return outputLayer().outputSize(); }

    private Layer inputLayer() { return layers.get(0); }
    private Layer outputLayer() { return layers.isEmpty() ? null : layers.get(layers.size() - 1); }

    public String toString(boolean withLastOutput) {
        return layers.stream()
                .map(layer -> layer.toString(withLastOutput))
                .collect(Collectors.joining("\n"));
    }

    @Override
    public String toString() {
        return toString(false);
    }
}
