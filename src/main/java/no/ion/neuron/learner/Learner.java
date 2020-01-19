package no.ion.neuron.learner;

import no.ion.neuron.tensor.Vector;

public interface Learner {
    /**
     * Information about an epoch that a learner can use to decide the adjustments to apply to the parameters..
     */
    class EpochInfo {
        private final float error;
        private final int batchSize;
        private final Vector gradient;

        public EpochInfo(float error, int batchSize, Vector gradient) {
            this.error = error;
            this.batchSize = batchSize;
            this.gradient = gradient;
        }

        /**
         * @return The total error over the training set, e.g. {@code sum_i (y_i - z_i)^2 /2}
         *         where the sum is over all examples in the training set, {@code y_i} is the
         *         computed output vector of the neural network, and {@code z_i} is the ideal
         *         output vector. Regularization may add additional terms to error. Aka E.
         */
        public float error() { return error; }

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
            return new EpochInfo(error, batchSize, gradient.copy());
        }
    }

    /** Calculate the adjustment to each parameter given information on a training set. */
    Vector learn(EpochInfo epochInfo);
}
