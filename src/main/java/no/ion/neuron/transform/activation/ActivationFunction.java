package no.ion.neuron.transform.activation;

/** A parameter-free transform that maps each input value x to an output value f(x). */
public interface ActivationFunction {
    /** y = f(x) */
    float f(float x);

    /** f'(x), with y = f(x). */
    float fGradient(float x, float y);
}
