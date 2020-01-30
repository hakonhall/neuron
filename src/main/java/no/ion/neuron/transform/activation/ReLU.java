package no.ion.neuron.transform.activation;

public class ReLU implements ActivationFunction {
    private final float epsilon;

    /**
     * @param epsilon e.g. 0.1
     */
    public ReLU(float epsilon) {
        this.epsilon = epsilon;
    }

    /** Plain ReLU */
    public ReLU() { this(0f); }

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
        return "ReLU{" + epsilon + '}';
    }
}
