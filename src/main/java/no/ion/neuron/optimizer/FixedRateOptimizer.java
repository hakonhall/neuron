package no.ion.neuron.optimizer;

import no.ion.neuron.tensor.Vector;

public class FixedRateOptimizer implements Optimizer {
    private final float learningRate;

    public FixedRateOptimizer(float learningRate) {
        this.learningRate = learningRate;

        if (learningRate <= 0) {
            throw new IllegalArgumentException("The learning rate must be positive: " + learningRate);
        }
    }

    @Override
    public Vector calculateParameterAdjustments(EpochSummary epochSummary) {
        Vector adjustments = epochSummary.gradientSum().copy();
        adjustments.multiplyScalar(-learningRate);
        return adjustments;
    }
}
