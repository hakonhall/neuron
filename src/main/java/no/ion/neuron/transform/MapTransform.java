package no.ion.neuron.transform;

import no.ion.neuron.tensor.Vector;
import no.ion.neuron.internal.BackPropagationImpl;
import no.ion.neuron.transform.mapper.Mapper;

/**
 * A transform built around a Mapper.
 */
public class MapTransform implements Transform {
    private final int size;
    private final Mapper mapper;

    public MapTransform(int size, Mapper mapper) {
        this.size = size;
        this.mapper = mapper;
    }

    @Override public int inputSize() { return size; }
    @Override public int outputSize() { return size; }
    @Override public int parameterSize() { return mapper.parameterSize(); }
    @Override public void adjustParameters(Vector amount) { mapper.adjustParameters(amount); }

    @Override
    public ComputationResult compute(Vector input, Vector idealOutput) {
        Vector output = new Vector(size);
        for (int i = 0; i < size; ++i) {
            float value = mapper.f(input.get(i));
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
                    float derivative = mapper.fGradient(input.get(i), output.get(i));
                    float errorGradientOfInput_i = derivative * errorGradientOfOutput.get(i);
                    errorGradientOfInput.addToElement(i, errorGradientOfInput_i);
                }

                Vector errorGradientOfParameter = new Vector(parameterSize());
                for (int j = 0; j < parameterSize(); ++j) {
                    for (int i = 0; i < outputSize(); ++i) {
                        float paramDerivative = mapper.computeGradientOfParameter(j, input.get(i), output.get(i));
                        errorGradientOfParameter.addToElement(j, errorGradientOfOutput.get(i) * paramDerivative);
                    }
                }

                return new BackPropagationImpl(errorGradientOfInput, errorGradientOfParameter);
            }
        };
    }

    @Override
    public String toString() {
        return "MapTransform{" +
                "mapper=" + mapper +
                '}';
    }
}
