package no.ion.neuron.transform;

import no.ion.neuron.tensor.Vector;
import no.ion.neuron.internal.BackPropagationImpl;

/**
 * Scale each input value with its own scaling parameter.
 * A special case of a WeightTransform - when there is no mixing.
 */
public class ScalingTransform implements Transform {
    private final Vector scales;

    public ScalingTransform(Vector scales) {
        this.scales = scales;
    }

    @Override public int inputSize() { return scales.size(); }
    @Override public int outputSize() { return scales.size(); }
    @Override public int parameterSize() { return scales.size(); }

    @Override
    public Vector compute(Vector input, Vector idealOutput) {
        Vector output = input.copy();
        output.scale(scales);
        return output;
    }

    @Override
    public BackPropagation backPropagate(Vector input, Vector output, Vector idealOutput, Vector errorGradientOfOutput) {
        BackPropagationImpl backPropagation = new BackPropagationImpl(inputSize(), parameterSize());

        for (int i = 0; i < inputSize(); ++i) {
            float gradientOfInput = errorGradientOfOutput.get(i) * scales.get(i);
            backPropagation.errorGradientOfInputs().setElement(i, gradientOfInput);

            float gradientOfParameter = errorGradientOfOutput.get(i) * input.get(i);
            backPropagation.errorGradientOfParameters().setElement(i, gradientOfParameter);
        }

        return backPropagation;
    }

    @Override
    public void adjustParameters(Vector amount) {
        scales.add(amount);
    }
}
