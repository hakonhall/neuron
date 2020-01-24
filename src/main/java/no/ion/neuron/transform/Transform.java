package no.ion.neuron.transform;

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

    /**
     * The result of computing the back-propagation, see {@link #backPropagate(Vector, Vector, Vector, Vector) backPropagate}
     * for an introduction.
     */
    interface BackPropagation {
        /** dE/dXj, where E is the error/loss, and Xj is the j'th input (input.get(j)). */
        Vector errorGradientOfInputs();

        /** dE/dPk, where E is the error/loss, and Pk is the k'th parameter. */
        Vector errorGradientOfParameters();
    }

    interface ComputationResult {
        Vector output();

        BackPropagation backPropagate(Vector errorGradientOfOutput);
    }

    default ComputationResult compute2(Vector input, Vector idealOutput) { return null; }

    /**
     * Compute the output vector given an input vector.
     *
     * @param input the input vector from the upstream layer.
     * @param idealOutput the ideal output of the neural network. null unless learning.
     * @return the output vector to pass to the downstream layer.
     */
    Vector compute(Vector input, Vector idealOutput);

    /**
     * The output of the forward computation of the neural network, for which this transform received the given
     * input and produced the given output, produced an error (or loss) E. The i'th element of
     * {@code errorGradientOfOutput} is {@code dE / dYi}, where {@code Yi} is the i'th component of the
     * {@code output} vector. I.e. it denotes the rate of change of E, per unit of change of the output.
     *
     * @param input                 The input vector to a forward computation.
     * @param output                The output vector from the forward computation, from a call to {@link #compute(Vector, Vector)}.
     * @param idealOutput
     * @param errorGradientOfOutput The error gradient w.r.t. the output vector.
     * @return the back-propagation result, see {@link BackPropagation}
     */
    BackPropagation backPropagate(Vector input, Vector output, Vector idealOutput, Vector errorGradientOfOutput);

    /**
     * @param amount the amount to adjust the parameters. The {@code amount} vector has parameters matching 1-1
     *               with the output of {@link BackPropagation#errorGradientOfParameters()}.
     */
    void adjustParameters(Vector amount);
}
