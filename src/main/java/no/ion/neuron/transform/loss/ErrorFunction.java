package no.ion.neuron.transform.loss;

import no.ion.neuron.tensor.Vector;

/**
 * In training a neural network, the parameters of the network will be adjusted to minimize the loss/error -
 * a float denoting the error of the computation: the smaller the number the better the neural network performs.
 *
 * <p>A fundamental assumption of this loss function is that computations of the neural net is independent -
 * the error of a single computation is independent of any other computations.</p>
 */
public interface LossFunction {
    interface Computation {
        /** The loss/error of the computation. */
        float error();

        /** dE / dy_i, where E is error() and y_i is output.get(i). */
        Vector errorGradientOfOutput();

        /** May be null if parameterSize() is 0. */
        default Vector errorGradientOfParameters() { return null; }
    }

    /** Compute the loss given the output of the neural network, and the ideal output of the neural network. */
    Computation compute(Vector output, Vector idealOutput);

    default int parameterSize() { return 0; }
    default void adjustParameters(Vector amount) {}
}
