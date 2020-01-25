package no.ion.neuron.transform;

import no.ion.neuron.internal.BackPropagationImpl;
import no.ion.neuron.tensor.Vector;
import no.ion.neuron.transform.loss.ErrorFunction;

public class OutputTransform implements Transform {
    private final int inputSize;
    private final ErrorFunction errorFunction;

    public OutputTransform(int inputSize, ErrorFunction errorFunction) {
        this.inputSize = inputSize;
        this.errorFunction = errorFunction;
    }

    @Override public int inputSize() { return inputSize; }
    @Override public int outputSize() { return 1; }
    @Override public int parameterSize() { return errorFunction.parameterSize(); }

    @Override
    public ComputationResult compute(Vector input, Vector idealOutput) {
        ErrorFunction.Computation computation = errorFunction.compute(input, idealOutput);
        return new ComputationResult() {
            @Override
            public Vector output() {
                return Vector.from(computation.error());
            }

            @Override
            public BackPropagation backPropagate(Vector errorGradient) {
                return new BackPropagationImpl(computation.errorGradientOfOutput(), computation.errorGradientOfParameters());
            }
        };
    }

    @Override
    public void adjustParameters(Vector amount) {
        errorFunction.adjustParameters(amount);
    }

    @Override
    public String toString() {
        return "OutputTransform{" +
                "inputSize=" + inputSize +
                ", errorFunction=" + errorFunction +
                '}';
    }
}
