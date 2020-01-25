package no.ion.neuron.transform.activation;

import static java.lang.Math.exp;

public class Sigmoid implements ActivationFunction {
    @Override
    public float f(float x) {
        return (float) (1 / (1 + exp(-x)));
    }

    @Override
    public float fGradient(float x, float y) {
        return y * (1 - y);
    }

    @Override
    public String toString() {
        return "Sigmoid{}";
    }
}
