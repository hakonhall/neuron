package no.ion.neuron.internal;

import no.ion.neuron.tensor.Vector;
import no.ion.neuron.optimizer.Optimizer;
import no.ion.neuron.optimizer.ComputationId;
import no.ion.neuron.optimizer.LayerId;
import no.ion.neuron.optimizer.ParametrizedLayer;
import no.ion.neuron.transform.Transform;

public class Layer implements ParametrizedLayer {
    private final LayerId layerId;
    private final Transform transform;
    private final Optimizer optimizer;

    private Layer upstream = null;
    private Layer downstream = null;

    private Vector lastOutput;

    public Layer(Transform transform, Optimizer optimizer) {
        this.layerId = LayerId.createNext();
        this.transform = transform;
        this.optimizer = optimizer;
    }

    @Override
    public LayerId layerId() { return layerId; }
    @Override
    public int parameterSize() { return transform.parameterSize(); }

    public void setUpstreamLayer(Layer upstream) { this.upstream = upstream; }
    public void setDownstreamLayer(Layer downstream) { this.downstream = downstream; }
    public int inputSize() { return transform.inputSize(); }
    public int outputSize() { return transform.outputSize(); }

    public static class Result {
        private final Vector netOutput;
        private Vector errorGradient;

        public Result(Vector netOutput) { this.netOutput = netOutput; }

        public Vector netOutput() { return netOutput; }
        public Vector errorGradient() { return errorGradient; }

        public Result setErrorGradient(Vector errorGradient) {
            this.errorGradient = errorGradient;
            return this;
        }
    }

    public Result process(ComputationId computationId, Vector input, Vector idealOutput) {
        Transform.ComputationResult result = transform.compute2(input, idealOutput);
        if (result != null) {
            Vector output = result.output();
            if (output.size() != transform.outputSize()) {
                throw new IllegalStateException("Output size from transform " + transform +
                        " does not match the declared output size: " + outputSize());
            }

            Result downstreamResult;
            if (this.downstream == null) {
                Vector errorGradient = new Vector(transform.outputSize(), 1f);
                downstreamResult = new Result(output).setErrorGradient(errorGradient);
            } else {
                downstreamResult = downstream.process(computationId, output, idealOutput);
            }

            Transform.BackPropagation backPropagation = result.backPropagate(downstreamResult.errorGradient);

            if (backPropagation.errorGradientOfInputs().size() != transform.inputSize()) {
                throw new IllegalStateException(String.format("Back-propagation vector size %d does not match transform's input size %d",
                        backPropagation.errorGradientOfInputs().size(), transform.inputSize()));
            }

            optimizer.registerErrorGradientOfParameters(computationId, this, backPropagation.errorGradientOfParameters());

            // Only error gradient different from downstream result.
            return downstreamResult.setErrorGradient(backPropagation.errorGradientOfInputs());
        }

        Vector output = transform.compute(input, idealOutput);
        lastOutput = output;

        Result downstreamResult;
        if (this.downstream == null) {
            Vector errorGradient = optimizer.errorGradient(computationId, output);
            downstreamResult = new Result(output).setErrorGradient(errorGradient);
        } else {
            downstreamResult = downstream.process(computationId, output, idealOutput);
        }

        Transform.BackPropagation backPropagation = transform.backPropagate(input, output, idealOutput, downstreamResult.errorGradient());

        if (backPropagation.errorGradientOfInputs().size() != transform.inputSize()) {
            throw new IllegalStateException(String.format("Back-propagation vector size %d does not match transform's input size %d",
                    backPropagation.errorGradientOfInputs().size(), transform.inputSize()));
        }

        optimizer.registerErrorGradientOfParameters(computationId, this, backPropagation.errorGradientOfParameters());

        return downstreamResult.setErrorGradient(backPropagation.errorGradientOfInputs());
    }

    @Override
    public void adjustParameters(Vector parameterAdjustments) {
        transform.adjustParameters(parameterAdjustments);
    }

    public String toString(boolean withLastOutput) {
        return "Layer{" +
                "layerId=" + layerId +
                ", transform=" + transform +
                (withLastOutput ? ", lastOutput=" + lastOutput : "") +
                '}';
    }

    @Override
    public String toString() {
        return toString(false);
    }
}
