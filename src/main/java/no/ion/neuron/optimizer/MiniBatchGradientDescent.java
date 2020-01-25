package no.ion.neuron.optimizer;

import no.ion.neuron.learner.FixedRateOptimizer;
import no.ion.neuron.learner.Optimizer;
import no.ion.neuron.tensor.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;
import java.util.stream.Collectors;

/**
 * Mini-batch gradient descent regularization that can be applied to a single layer.
 */
public class MiniBatchGradientDescent implements GradientGatherer {
    private final int batchSize;
    private final TreeMap<ComputationId, Computation> computations = new TreeMap<>();
    private final Optimizer optimizer;

    /** The current epoch's layers. */
    private TreeMap<LayerId, LayerInfo> layers = new TreeMap<>();
    /** Contains the previous epoch's layers. */
    private TreeMap<LayerId, LayerInfo> previousLayers = null;

    private int computationsCompletedInBatch = 0;
    private int ongoingComputations = 0;

    private Vector lastOutputSum = null;
    private Vector outputSum = null;

    /**
     * A: The distance to the component of dP_{n+1} along dP_n, negative if backwards.
     * B: The length of the transverse component of dP_{n+1} compared to dP_n.
     */
    private ArrayList<Float> As = null;
    private ArrayList<Float> Bs = null;

    public MiniBatchGradientDescent(int batchSize, Optimizer optimizer) {
        this.batchSize = batchSize;
        this.optimizer = optimizer;
    }

    public MiniBatchGradientDescent(int batchSize, float epsilon) {
        this(batchSize, new FixedRateOptimizer(epsilon));
    }

    @Override
    public void startComputation(ComputationId computationId, Vector input, Vector idealOutput) {
        if (ongoingComputations > 0) {
            throw new IllegalStateException(getClass().getSimpleName() + " does not support concurrent computations");
        }

        ++ongoingComputations;
        computations.put(computationId, new Computation(input, idealOutput));
    }

    @Override
    public void registerErrorGradientOfParameters(ComputationId computationId, ParametrizedLayer layer, Vector errorGradient) {
        if (layer.parameterSize() != errorGradient.size()) {
            throw new IllegalArgumentException("The number of parameters (" + layer.parameterSize() +
                    ") in layer " + layer.layerId() + " does not match the size of the error gradient (w.r.t. parameters) "
                    + errorGradient.size());
        }

        LayerInfo layerInfo = layers.get(layer.layerId());
        if (layerInfo == null) {
            layerInfo = new LayerInfo(layer, errorGradient.copy());
            layers.put(layer.layerId(), layerInfo);
        } else {
            assert layerInfo.layer == layer;
            layerInfo.cumulativeGradient.add(errorGradient);
        }
    }

    @Override
    public void endComputation(ComputationId computationId, Vector output) {
        if (outputSum == null) {
            outputSum = output.copy();
        } else {
            outputSum.add(output);
        }
        --ongoingComputations;
        ++computationsCompletedInBatch;
        if (computationsCompletedInBatch < batchSize) {
            return;
        }

        List<Vector> gradients = layers.values().stream().map(info -> info.cumulativeGradient).collect(Collectors.toList());
        int numParams = gradients.stream().map(Vector::size).reduce(Integer::sum).orElse(0);

        Vector unifiedGradient = new Vector(numParams);
        int i = 0;
        for (var gradient : gradients) {
            for (int j = 0; j < gradient.size(); ++j) {
                unifiedGradient.setElement(i, gradient.get(j));
                ++i;
            }
        }

        Optimizer.EpochInfo epochInfo = new Optimizer.EpochInfo(outputSum.copy(), batchSize, unifiedGradient);
        Vector adjustments = optimizer.learn(epochInfo);

        i = 0;
        for (var layerInfo : layers.values()) {
            Vector layerAdjustments = new Vector(layerInfo.cumulativeGradient.size());
            for (int j = 0; j < layerAdjustments.size(); ++j) {
                layerAdjustments.setElement(j, adjustments.get(i));
                ++i;
            }

            layerInfo.layer.adjustParameters(layerAdjustments);
        }

        layers.forEach((id, info) -> info.cumulativeGradient.clear());
        computations.remove(computationId);

        computationsCompletedInBatch = 0;
        lastOutputSum = outputSum.copy();
        outputSum.clear();
    }

    public Vector lastOutputSum() { return lastOutputSum; }
    public int batchSize() { return batchSize; }

    private static class Computation {
        private final Vector input;
        private final Vector idealOutput;

        public Computation(Vector input, Vector idealOutput) {
            this.input = input;
            this.idealOutput = idealOutput;
        }
    }

    private static class LayerInfo {
        private final ParametrizedLayer layer;
        private Vector cumulativeGradient;

        public LayerInfo(ParametrizedLayer layer, Vector initialGradient) {
            this.layer = layer;
            this.cumulativeGradient = initialGradient;
        }

        /** Is destructive to the data of this instance. */
        public void applyAndClear(float factor) {
            cumulativeGradient.multiplyScalar(factor);
            // System.out.println("Applying adjustments on layer " + layer.layerId() + ": " + cumulativeGradient);
            layer.adjustParameters(cumulativeGradient);
            cumulativeGradient = new Vector(cumulativeGradient.size());
        }
    }
}
