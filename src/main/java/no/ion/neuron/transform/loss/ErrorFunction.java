package no.ion.neuron.transform.loss;

import no.ion.neuron.tensor.Vector;

/**
 * In training a neural network, the parameters of the network will be adjusted to minimize the error/loss -
 * a float denoting the error of the computation: the smaller the number the better the neural network performs.
 *
 * <p>A fundamental assumption of this error function is that computations of the neural net are independent -
 * the error of a single computation is independent of any other computations.</p>
 */
public interface ErrorFunction {
    interface Computation {
        /** The error/loss of the computation. */
        float error();

        /** dE/dYj, where E is error() and Yj is output.get(j). */
        Vector errorGradientOfOutput();

        /** dE/dPk, where E is error(), and Pk is the k'th parameter of the error function. */
        default Vector errorGradientOfParameters() { return new Vector(0); }
    }

    /** Compute the error/loss given the output of the neural network, and the ideal output of the neural network. */
    Computation compute(Vector output, Vector idealOutput);

    default int parameterSize() { return 0; }
    default void adjustParameters(Vector amount) {}
    String toString();
}
