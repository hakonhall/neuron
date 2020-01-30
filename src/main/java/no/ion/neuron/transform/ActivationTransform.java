package no.ion.neuron.transform;

import no.ion.neuron.ComputeContext;
import no.ion.neuron.internal.BackPropagationImpl;
import no.ion.neuron.tensor.Vector;
import no.ion.neuron.transform.activation.ActivationFunction;

/**
 * A transform built around an ActivationFunction.
 */
public class ActivationTransform implements Transform {
    private final int size;
    private final ActivationFunction activationFunction;

    public ActivationTransform(int size, ActivationFunction activationFunction) {
        this.size = size;
        this.activationFunction = activationFunction;
    }

    @Override public int inputSize() { return size; }
    @Override public int outputSize() { return size; }
    @Override public int parameterSize() { return 0; }
    @Override public void adjustParameters(Vector amount) { }
    @Override public Vector parameters() { return new Vector(0); }

    @Override
    public ComputationResult compute(ComputeContext context, Vector input) {
        Vector output = new Vector(size);
        for (int i = 0; i < size; ++i) {
            float value = activationFunction.f(input.get(i));
            output.setElement(i, value);
        }

        return new ComputationResult() {
            @Override
            public Vector output() {
                return output;
            }

            @Override
            public BackPropagation backPropagate(Vector errorGradientOfOutput) {
                Vector errorGradientOfInput = new Vector(inputSize());
                for (int i = 0; i < inputSize(); ++i) {
                    float derivative = activationFunction.fGradient(input.get(i), output.get(i));
                    float errorGradientOfInput_i = derivative * errorGradientOfOutput.get(i);
                    errorGradientOfInput.addToElement(i, errorGradientOfInput_i);
                }

                Vector errorGradientOfParameter = new Vector(0);

                return new BackPropagationImpl(errorGradientOfInput, errorGradientOfParameter);
            }
        };
    }

    @Override
    public String toString() {
        return "ActivationTransform{" +
                "activationFunction=" + activationFunction +
                '}';
    }
}
