package no.ion.neuron.transform;

import no.ion.neuron.internal.BackPropagationImpl;
import no.ion.neuron.tensor.Vector;

public class OutputTransform implements Transform {
    private final int inputSize;

    public OutputTransform(int inputSize) {
        this.inputSize = inputSize;
    }

    @Override public int inputSize() { return inputSize; }
    @Override public int outputSize() { return 1; }
    @Override public int parameterSize() { return 0; }

    @Override
    public Vector compute(Vector input, Vector idealOutput) {
        Vector error = input.copy();
        error.subtract(idealOutput);
        float E = error.squared() / 2;
        return Vector.from(E);
    }

    @Override
    public BackPropagation backPropagate(Vector input, Vector output, Vector idealOutput, Vector errorGradientOfOutput) {
        BackPropagation backPropagation = new BackPropagationImpl(inputSize, parameterSize());

        for (int j = 0; j < inputSize; ++j) {
            float diff = input.get(j) - idealOutput.get(j);
            backPropagation.errorGradientOfInputs().setElement(j, errorGradientOfOutput.get(0) * diff);
        }

        return backPropagation;
    }

    @Override
    public void adjustParameters(Vector amount) {
        if (amount.size() > 0) {
            throw new IllegalArgumentException("OutputTransform has no parameters");
        }
    }
}
