package no.ion.neuron.transform.loss;

import no.ion.neuron.tensor.Vector;

/**
 * A loss function being (half) the square of the error vector,
 * the error vector being the difference between the output vector with the ideal output vector.
 */
public class ErrorSquared implements LossFunction {
    @Override
    public Computation compute(Vector output, Vector idealOutput) {
        Vector error = output.copy();
        error.subtract(idealOutput);

        return new Computation() {
            @Override
            public float error() {
                return error.squared() / 2f;
            }

            @Override
            public Vector errorGradientOfOutput() {
                return error;
            }
        };
    }
}
