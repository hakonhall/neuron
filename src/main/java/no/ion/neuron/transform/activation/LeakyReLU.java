package no.ion.neuron.transform.activation;

public class LeakyReLU implements ActivationFunction {
    private final float epsilon;

    /**
     * @param epsilon e.g. 0.1
     */
    public LeakyReLU(float epsilon) {
        this.epsilon = epsilon;
    }

    /** Plain ReLU */
    public LeakyReLU() { this(0f); }

    @Override
    public float f(float x) {
        return x < 0 ? epsilon * x : x;
    }

    @Override
    public float fGradient(float x, float y) {
        return x < 0 ? epsilon : 1;
    }

    @Override
    public String toString() {
        return "LeakyReLU{" + epsilon + '}';
    }
}
