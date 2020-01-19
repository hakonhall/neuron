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
     * Compute the output vector given an input vector.
     *
     * @param input the input vector from the upstream layer.
     * @param idealOutput the ideal output of the neural network. null unless learning.
     * @return the output vector to pass to the downstream layer.
     */
    Vector compute(Vector input, Vector idealOutput);

    /**
     * The result of computing the back-propagation, see {@link #backPropagate(Vector, Vector, Vector, Vector) backPropagate}
     * for an introduction.
     */
    interface BackPropagation {
        /**
         * @return a vector of size {@link Transform#inputSize()}, with the j'th element being dE / dXj,
         *         where Xj is the j'th element of the {@code input} vector, and hence the j'th element
         *         of the output vector of the upstream layer, thereby providing the upstream layer with
         *         it's {@code errorGradientOfOutput} necessary to apply its back-propagation. Note that
         *         {@code dE / dXj = sum_i dE/dYi dYi/dXj}, where {@code dE/dYi} is the errorGradientOfOutput
         *         passed to {@link #backPropagate(Vector, Vector, Vector, Vector)}. Therefore the only unknowns
         *         this function needs to find are the {@code dYi/dXj}.
         */
        Vector errorGradientOfInputs();

        /**
         * @return the error gradient w.r.t. the parameters of the transform, i.e. {@code dE / dPj}
         *         where Pj is the j'th learnable (adjustable) internal parameter of this transform.
         *         The number of parameters is determined by the transform. Note that
         *         {@code dE / dPj = sum_i dE/dYi dYi/dPj}, where {@code dE/dYi} is the errorGradientOfOutput
         *         passed to {@link #backPropagate(Vector, Vector, Vector, Vector)}. Therefore the only unknowns
         *         this function needs to find are {@code dYi/dPj}.
         */
        Vector errorGradientOfParameters();
    }

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
