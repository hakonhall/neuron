package no.ion.neuron.learner;

import no.ion.neuron.tensor.Vector;

public class FixedRateLearner implements Learner {
    private final float learningRate;

    public FixedRateLearner(float learningRate) {
        this.learningRate = learningRate;

        if (learningRate <= 0) {
            throw new IllegalArgumentException("The learning rate must be positive: " + learningRate);
        }
    }

    @Override
    public Vector learn(EpochInfo epochInfo) {
        Vector adjustments = epochInfo.gradient().copy();
        adjustments.multiplyScalar(-learningRate);
        return adjustments;
    }
}
