package no.ion.neuron.transform;

import no.ion.neuron.ComputeContext;
import no.ion.neuron.tensor.Matrix;
import no.ion.neuron.tensor.Vector;
import no.ion.neuron.internal.BackPropagationImpl;

public class WeightTransform implements Transform {
    private final Matrix weight;

    public WeightTransform(Matrix weight) {
        this.weight = weight;
    }

    public Matrix weight() { return weight; }
    @Override public int inputSize() { return weight.columns(); }
    @Override public int outputSize() { return weight.rows(); }
    @Override public int parameterSize() { return weight.rows() * weight.columns(); }
    @Override public Vector parameters() { return weight.toVector(); }

    @Override
    public ComputationResult compute(ComputeContext context, Vector input) {
        Vector output = weight.dot(input);

        return new ComputationResult() {
            @Override
            public Vector output() {
                return output;
            }

            @Override
            public BackPropagation backPropagate(Vector errorGradientOfOutput) {
                // dE/dXj = sum_i dE/dYi * Wij
                Vector errorGradientOfInputs = new Vector(inputSize());

                // dE/dWij = dE/dyi * xj
                Matrix errorGradientOfParameters = new Matrix(outputSize(), inputSize());

                for (int j = 0; j < inputSize(); ++j) {
                    for (int i = 0; i < outputSize(); ++i) {
                        errorGradientOfInputs.addToElement(j, errorGradientOfOutput.get(i) * weight.getElement(i, j));
                        errorGradientOfParameters.setElement(i, j, errorGradientOfOutput.get(i) * input.get(j));
                    }
                }

                return new BackPropagationImpl(errorGradientOfInputs, errorGradientOfParameters.toVector());
            }
        };
    }

    @Override
    public void adjustParameters(Vector amount) {
        weight.add(amount.toMatrix(outputSize()));
    }

    @Override
    public String toString() {
        return "WeightTransform{" +
                "weight=" + weight +
                '}';
    }
}
