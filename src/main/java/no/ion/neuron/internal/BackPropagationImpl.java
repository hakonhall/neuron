package no.ion.neuron.internal;

import no.ion.neuron.tensor.Vector;
import no.ion.neuron.transform.Transform;

public class BackPropagationImpl implements Transform.BackPropagation {
    private final Vector errorGradientOfInputs;
    private final Vector errorGradientOfParameters;

    public BackPropagationImpl(Vector errorGradientOfInputs, Vector errorGradientOfParameters) {
        this.errorGradientOfInputs = errorGradientOfInputs;
        this.errorGradientOfParameters = errorGradientOfParameters;
    }

    public BackPropagationImpl(int inputSize, int parameterSize) {
        this(new Vector(inputSize), new Vector(parameterSize));
    }

    @Override public Vector errorGradientOfInputs() { return errorGradientOfInputs; }
    @Override public Vector errorGradientOfParameters() { return errorGradientOfParameters; }
}
