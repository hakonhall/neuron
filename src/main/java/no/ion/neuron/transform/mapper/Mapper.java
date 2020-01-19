package no.ion.neuron.transform.mapper;

import no.ion.neuron.tensor.Vector;

/** A mapper maps one input value to one output value, and may depend on tunable parameters. */
public interface Mapper {
    /** y = f(x) */
    float f(float x);

    /** fgradient(x, y) == f'(x), with y = f(x). */
    float fGradient(float x, float y);

    default int parameterSize() { return 0; }

    default float computeGradientOfParameter(int parameterIndex, float value, float computedValue) {
        throw new UnsupportedOperationException();
    }

    default void adjustParameters(Vector amount) {
        if (amount.size() > 0) {
            throw new UnsupportedOperationException();
        }
    }
}
