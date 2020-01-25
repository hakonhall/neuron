package no.ion.neuron.transform.activation;

import static java.lang.Math.tanh;

public class Tanh implements ActivationFunction {
    @Override
    public float f(float x) {
        return (float) tanh(x);
    }

    @Override
    public float fGradient(float x, float y) {
        return 1 - y * y;
    }

    @Override
    public String toString() {
        return "Tanh{}";
    }
}
