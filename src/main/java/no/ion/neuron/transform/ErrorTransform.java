package no.ion.neuron.transform;

import no.ion.neuron.ComputeContext;
import no.ion.neuron.internal.BackPropagationImpl;
import no.ion.neuron.tensor.Vector;
import no.ion.neuron.transform.Transform;
import no.ion.neuron.transform.loss.ErrorFunction;

/**
 * Makes a Transform out of an ErrorFunction.
 */
public class ErrorTransform implements Transform {
    private final int inputSize;
    private final ErrorFunction errorFunction;

    public ErrorTransform(int inputSize, ErrorFunction errorFunction) {
        this.inputSize = inputSize;
        this.errorFunction = errorFunction;
    }

    @Override public int inputSize() { return inputSize; }
    @Override public int outputSize() { return 1; }
    @Override public int parameterSize() { return errorFunction.parameterSize(); }
    @Override public Vector parameters() { return errorFunction.parameters(); }

    @Override
    public ComputationResult compute(ComputeContext context, Vector input) {
        ErrorFunction.Computation computation = errorFunction.compute(input, context.idealOutput());
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
        return "ErrorTransform{" + errorFunction + '}';
    }
}
