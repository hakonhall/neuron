package no.ion.neuron.transform.activation;

import static java.lang.Math.log;
import static java.lang.Math.sqrt;

public class Asinh implements ActivationFunction {
    @Override
    public float f(float x) {
        return x < 0 ? -f(-x) : (float) log(x + sqrt(x * x + 1));
    }

    @Override
    public float fGradient(float x, float y) {
        return (float) sqrt(x * x + 1);
    }

    @Override
    public String toString() {
        return "Asinh{}";
    }
}
