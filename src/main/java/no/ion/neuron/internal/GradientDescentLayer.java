package no.ion.neuron.internal;

import no.ion.neuron.ComputeContext;
import no.ion.neuron.tensor.Vector;
import no.ion.neuron.optimizer.LayerId;
import no.ion.neuron.optimizer.ParametrizedLayer;
import no.ion.neuron.transform.Transform;

public class GradientDescentLayer implements ParametrizedLayer {
    private final LayerId layerId;
    private final Transform transform;
    private final Vector cumulativeErrorGradientOfParameters;

    private GradientDescentLayer upstream = null;
    private GradientDescentLayer downstream = null;

    public GradientDescentLayer(Transform transform) {
        this.layerId = LayerId.createNext();
        this.transform = transform;
        this.cumulativeErrorGradientOfParameters = new Vector(transform.parameterSize());
    }

    @Override
    public LayerId layerId() { return layerId; }

    public void setUpstreamLayer(GradientDescentLayer upstream) { this.upstream = upstream; }
    public void setDownstreamLayer(GradientDescentLayer downstream) { this.downstream = downstream; }
    public int inputSize() { return transform.inputSize(); }
    public int outputSize() { return transform.outputSize(); }

    public static class ProcessResult {
        private final Vector netOutput;
        private Vector errorGradient;

        public ProcessResult(Vector netOutput) { this.netOutput = netOutput; }

        public Vector netOutput() { return netOutput; }
        public Vector errorGradient() { return errorGradient; }

        public ProcessResult setErrorGradient(Vector errorGradient) {
            this.errorGradient = errorGradient;
            return this;
        }
    }

    public ProcessResult process(ComputeContext context, Vector input) {
        Transform.ComputationResult result = transform.compute(context, input);

        Vector output = result.output();
        if (output.size() != transform.outputSize()) {
            throw new IllegalStateException("Output size from transform " + transform +
                    " does not match the declared output size: " + outputSize());
        }

        ProcessResult downstreamResult;
        if (this.downstream == null) {
            Vector errorGradient = new Vector(transform.outputSize(), 1f);
            downstreamResult = new ProcessResult(output).setErrorGradient(errorGradient);
        } else {
            downstreamResult = downstream.process(context, output);
        }

        Transform.BackPropagation backPropagation = result.backPropagate(downstreamResult.errorGradient);

        if (backPropagation.errorGradientOfInputs().size() != transform.inputSize()) {
            throw new IllegalStateException(String.format("Back-propagation vector size %d does not match transform's input size %d",
                    backPropagation.errorGradientOfInputs().size(), transform.inputSize()));
        }

        Vector errorGradientOfParameters = backPropagation.errorGradientOfParameters();
        cumulativeErrorGradientOfParameters.add(errorGradientOfParameters);

        // Only error gradient different from downstream result.
        return downstreamResult.setErrorGradient(backPropagation.errorGradientOfInputs());
    }

    @Override public int parameterSize() { return transform.parameterSize(); }
    @Override public Vector cumulativeErrorGradientOfParameters() { return cumulativeErrorGradientOfParameters; }
    @Override public void clearCumulativeErrorGradientOfParameters() { cumulativeErrorGradientOfParameters.clear(); }
    @Override public void adjustParameters(Vector parameterAdjustments) { transform.adjustParameters(parameterAdjustments); }

    public String toString(boolean withLastOutput) {
        return "GradientDescentLayer{" +
                "layerId=" + layerId +
                ", transform=" + transform +
                '}';
    }

    @Override
    public String toString() {
        return toString(false);
    }
}
