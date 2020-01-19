package no.ion.neuron.transform.mapper;

import static java.lang.Math.tanh;

public class Tanh implements Mapper {
    public Tanh() {
    }

    @Override
    public float f(float x) {
        return (float) tanh(x);
    }

    @Override
    public float fGradient(float x, float y) {
        return 1 - y * y;
    }
}
