package no.ion.neuron.transform;

import no.ion.neuron.ComputeContext;
import no.ion.neuron.tensor.Vector;

/**
 * A computation atom in a feed-forward neural network, supporting the computation of
 * an output vector given an input vector, and the back-propagation of the error gradient
 * in the opposite direction for learning (adjusting) parameters.
 */
public interface Transform {

    /** The size of the input vector. */
    int inputSize();

    /** The size of the output vector. */
    int outputSize();

    /** The number of parameters. */
    int parameterSize();

    /** The current value of the parameters. */
    Vector parameters();

    interface BackPropagation {
        /** dE/dXj = sum_i dE/dYi dYi/dXj, where E is the error/loss, and Xj is the j'th input (input.get(j)). */
        Vector errorGradientOfInputs();

        /** dE/dPk = sum_i dE/dYi dYi/dPk, where E is the error/loss, and Pk is the k'th parameter. */
        Vector errorGradientOfParameters();
    }

    interface ComputationResult {
        /** The output of the transformation. */
        Vector output();

        /**
         * Invoked once the error gradient of the output is known, to calculate the effect of back-propagation.
         *
         * @param errorGradientOfOutput dE/dYi, where E is the error/loss, and Yi is the i'th output (output().get(i)).
         */
        BackPropagation backPropagate(Vector errorGradientOfOutput);
    }

    ComputationResult compute(ComputeContext context, Vector input);

    /**
     * @param amount the amount to adjust the parameters. The {@code amount} vector has parameters matching 1-1
     *               with the output of {@link BackPropagation#errorGradientOfParameters()}.
     */
    void adjustParameters(Vector amount);
}
