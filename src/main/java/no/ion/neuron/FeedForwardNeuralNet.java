package no.ion.neuron;

import no.ion.neuron.internal.GradientDescentLayer;
import no.ion.neuron.optimizer.ParametrizedLayer;
import no.ion.neuron.tensor.Matrix;
import no.ion.neuron.tensor.Vector;
import no.ion.neuron.transform.BiasTransform;
import no.ion.neuron.transform.IdentityTransform;
import no.ion.neuron.transform.MapTransform;
import no.ion.neuron.transform.Transform;
import no.ion.neuron.transform.WeightTransform;
import no.ion.neuron.transform.mapper.Mapper;

import java.lang.reflect.ParameterizedType;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class FeedForwardNeuralNet {
    private final List<GradientDescentLayer> layers = new ArrayList<>();

    public FeedForwardNeuralNet(int inputSize) {
        layers.add(new GradientDescentLayer(new IdentityTransform(inputSize)));
    }

    public GradientDescentLayer addLayer(Matrix weight, Vector bias, Mapper mapper) {
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

    public GradientDescentLayer addTransform(Transform transform) {
        GradientDescentLayer layer = new GradientDescentLayer(transform);

        if (outputLayer() != null) {
            layer.setUpstreamLayer(outputLayer());
            outputLayer().setDownstreamLayer(layer);
        }

        layers.add(layer);
        return layer;
    }

    public int inputSize() { return inputLayer().inputSize(); }
    public int outputSize() { return outputLayer().outputSize(); }
    public List<? extends ParametrizedLayer> layers() { return List.copyOf(layers); }

    public Vector compute(Vector input, Vector idealOutput) {
        ComputeContext context = new ComputeContext(input);
        return compute(context, input);
    }

    public Vector compute(ComputeContext context, Vector input) {
        return inputLayer().process(context, input).netOutput();
    }

    public Vector gradientOfParameters() {
        int parameterSize = layers.stream().map(layer -> layer.parameterSize()).reduce(Integer::sum).orElse(0);
        Vector gradient = new Vector(parameterSize);

        int i = 0;
        for (var layer : layers) {
            Vector cumulativeErrorGradientOfParametersInLayer = layer.cumulativeErrorGradientOfParameters();
            for (int j = 0; j < cumulativeErrorGradientOfParametersInLayer.size(); ++j) {
                gradient.setElement(i++, cumulativeErrorGradientOfParametersInLayer.get(j));
            }
        }

        return gradient;
    }

    public void adjustParameters(Vector delta) {
        int i = 0;
        for (var layer : layers) {
            Vector adjustment = new Vector(layer.parameterSize());
            for (int j = 0; j < adjustment.size(); ++j) {
                float nextValue = delta.get(i++);
                adjustment.setElement(j, nextValue);
            }
            layer.adjustParameters(adjustment);
        }
    }

    public void clearCumulativeErrorGradientOfParameters() {
        layers.forEach(ParametrizedLayer::clearCumulativeErrorGradientOfParameters);
    }

    private GradientDescentLayer inputLayer() { return layers.get(0); }
    private GradientDescentLayer outputLayer() { return layers.get(layers.size() - 1); }

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
