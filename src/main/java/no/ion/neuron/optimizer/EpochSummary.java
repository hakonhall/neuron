package no.ion.neuron.optimizer;

import no.ion.neuron.tensor.Vector;

/**
 * Information about an epoch that a learner can use to decide the adjustments to apply to the parameters.
 */
public class EpochSummary {
    private final int epochs;
    private final float errorSum;
    private final int batchSize;
    private final Vector gradientSum;

    public EpochSummary(int epochs, float errorSum, int batchSize, Vector gradientSum) {
        this.epochs = epochs;
        this.errorSum = errorSum;
        this.batchSize = batchSize;
        this.gradientSum = gradientSum;
    }

    /** The number of epochs, this including. */
    public int epochs() { return epochs; }

    /** E, the sum of the error of each example in the training set of the epoch. */
    public float errorSum() { return errorSum; }

    /** The number of examples in the training set. */
    public int batchSize() { return batchSize; }

    /** The number of parameters. */
    public int parameterSize() { return gradientSum.size(); }

    /** dE/dPk, where Pk is the k'th parameter of the neural network. */
    public Vector gradientSum() { return gradientSum; }

    public EpochSummary deepCopy() {
        return new EpochSummary(epochs, errorSum, batchSize, gradientSum.copy());
    }
}
