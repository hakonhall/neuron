package no.ion.neuron.transform.mapper;

import static java.lang.Math.log;
import static java.lang.Math.sqrt;

public class Arcsinh implements Mapper {
    public Arcsinh() {
    }

    @Override
    public float f(float x) {
        if (x < 0) {
            return -f(-x);
        }

        return (float) log(x + sqrt(x * x + 1));
    }

    @Override
    public float fGradient(float x, float y) {
        return (float) sqrt(x * x + 1);
    }
}
