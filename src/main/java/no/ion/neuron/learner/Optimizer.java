package no.ion.neuron.learner;

import no.ion.neuron.tensor.Vector;

public interface Optimizer {
    /**
     * Information about an epoch that a learner can use to decide the adjustments to apply to the parameters.
     */
    class EpochInfo {
        private final float errorSum;
        private final int batchSize;
        private final Vector gradient;

        public EpochInfo(float errorSum, int batchSize, Vector gradient) {
            this.errorSum = errorSum;
            this.batchSize = batchSize;
            this.gradient = gradient;
        }

        /** The sum of the output of the epoch. */
        public float errorSum() { return errorSum; }

        /** The number of examples in the training set. */
        public int batchSize() { return batchSize; }

        /** The number of parameters. */
        public int parameterSize() { return gradient.size(); }

        /**
         * @return The error grading w.r.t each parameter, i.e. {@code gradient.get(i)} is
         *         the rate of change of {@code error} w.r.t. parameter {@code i} ({@code dE/dP_i}).
         */
        public Vector gradient() { return gradient; }

        public EpochInfo deepCopy() {
            return new EpochInfo(errorSum, batchSize, gradient.copy());
        }
    }

    /** Calculate the adjustment to each parameter given information on a training set. */
    Vector calculateParameterAdjustments(EpochInfo epochInfo);
}
