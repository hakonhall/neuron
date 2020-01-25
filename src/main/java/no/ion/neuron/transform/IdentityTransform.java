package no.ion.neuron.transform;

import no.ion.neuron.tensor.Vector;
import no.ion.neuron.internal.BackPropagationImpl;

public class IdentityTransform implements Transform {
    private final int size;

    public IdentityTransform(int size) { this.size = size; }
    @Override public int inputSize() { return size; }
    @Override public int outputSize() { return size; }
    @Override public int parameterSize() { return 0; }

    @Override
    public ComputationResult compute(Vector input, Vector idealOutput) {
        return new ComputationResult() {
            @Override
            public Vector output() {
                return input;
            }

            @Override
            public BackPropagation backPropagate(Vector errorGradientOfOutput) {
                return new BackPropagationImpl(errorGradientOfOutput, Vector.from());
            }
        };
    }

    @Override
    public void adjustParameters(Vector amount) {
        if (amount.size() != 0) {
            throw new IllegalArgumentException();
        }
    }

    @Override
    public String toString() {
        return "IdentityTransform{" +
                "size=" + size +
                '}';
    }
}
