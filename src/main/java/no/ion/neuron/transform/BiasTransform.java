package no.ion.neuron.transform;

import no.ion.neuron.tensor.Vector;
import no.ion.neuron.internal.BackPropagationImpl;

public class BiasTransform implements Transform {
    private final Vector bias;

    public BiasTransform(Vector bias) {
        this.bias = bias;
    }

    public Vector bias() { return bias; }

    @Override public int inputSize(){ return bias.size(); }
    @Override public int outputSize() { return bias.size(); }
    @Override public int parameterSize() { return bias.size(); }

    @Override
    public Vector compute(Vector input, Vector idealOutput) {
        Vector output = input.copy();
        output.add(bias);
        return output;
    }

    @Override
    public BackPropagation backPropagate(Vector input, Vector output, Vector idealOutput, Vector errorGradientOfOutput) {
        Vector errorGradientOfInput = new Vector(inputSize());
        Vector errorGradientOfParameters = new Vector(parameterSize());
        for (int i = 0; i < bias.size(); ++i) {
            errorGradientOfInput.setElement(i, errorGradientOfOutput.get(i));
            errorGradientOfParameters.setElement(i, errorGradientOfOutput.get(i));
        }

        return new BackPropagationImpl(errorGradientOfInput, errorGradientOfParameters);
    }

    @Override
    public void adjustParameters(Vector amount) {
        bias.add(amount);
    }

    @Override
    public String toString() {
        return "BiasTransform{" +
                "bias=" + bias +
                '}';
    }
}
