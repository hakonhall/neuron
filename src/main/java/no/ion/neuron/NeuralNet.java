package no.ion.neuron;

import no.ion.neuron.gradientdescent.GradientDescentLayer;
import no.ion.neuron.layer.ParametrizedLayer;
import no.ion.neuron.tensor.Matrix;
import no.ion.neuron.tensor.Vector;
import no.ion.neuron.transform.ActivationTransform;
import no.ion.neuron.transform.BiasTransform;
import no.ion.neuron.transform.Transform;
import no.ion.neuron.transform.WeightTransform;
import no.ion.neuron.transform.activation.ActivationFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * A feed-forward neural network with back-propagation of gradients.
 */
public class NeuralNet {
    private final List<GradientDescentLayer> layers = new ArrayList<>();
    private final int inputSize;

    public NeuralNet(int inputSize) {
        this.inputSize = inputSize;
    }

    public void addLayers(Matrix weight, Vector bias, ActivationFunction activationFunction) {
        if (weight.columns() != outputSize()) {
            throw new IllegalArgumentException("Last layer has size " + outputSize() +
                    " but weight has input size " + weight.columns());
        } else if (bias.size() != weight.rows()) {
            throw new IllegalArgumentException("Weight output size " + weight.rows() +
                    " does not match bias size " + bias.size());
        }

        addTransform(new WeightTransform(weight));
        addTransform(new BiasTransform(bias));
        addTransform(new ActivationTransform(bias.size(), activationFunction));
    }

    public GradientDescentLayer addTransform(Transform transform) {
        GradientDescentLayer layer = new GradientDescentLayer(transform);

        if (layers.size() > 0) {
            GradientDescentLayer lastLayer = layers.get(layers.size() - 1);
            lastLayer.setDownstreamLayer(layer);
            layer.setUpstreamLayer(lastLayer);
        }
        layers.add(layer);
        return layer;
    }

    public int inputSize() { return inputSize; }
    public int outputSize() { return layers.isEmpty() ? inputSize : layers.get(layers.size() - 1).outputSize(); }
    public List<? extends ParametrizedLayer> layers() { return List.copyOf(layers); }

    public Vector compute(Vector input, Vector idealOutput) {
        ComputeContext context = new ComputeContext(input);
        return compute(context, input);
    }

    public Vector compute(ComputeContext context, Vector input) {
        if (layers.isEmpty()) {
            return input;
        } else {
            return layers.get(0).process(context, input).netOutput();
        }
    }

    /** The cumulative gradient w.r.t the parameters. */
    public Vector cumulativeGradientOfParameters() {
        return parameterVector(ParametrizedLayer::cumulativeErrorGradientOfParameters);
    }

    public Vector parameters() {
        return parameterVector(ParametrizedLayer::parameters);
    }

    private Vector parameterVector(Function<ParametrizedLayer, Vector> layerMethod) {
        int parameterSize = layers.stream().map(layer -> layer.parameterSize()).reduce(Integer::sum).orElse(0);
        Vector combinedVector = new Vector(parameterSize);

        int i = 0;
        for (var layer : layers) {
            Vector layerVector = layerMethod.apply(layer);
            for (int j = 0; j < layerVector.size(); ++j) {
                combinedVector.setElement(i++, layerVector.get(j));
            }
        }

        return combinedVector;
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
