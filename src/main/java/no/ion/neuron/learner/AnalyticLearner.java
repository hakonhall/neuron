package no.ion.neuron.learner;

import no.ion.neuron.tensor.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Wraps a learner and exposes various metrics that can be used to analyse the wrapped learner.
 */
public class AnalyticLearner implements Learner {
    private final Learner wrappedLearner;
    private final List<Epoch> epochs = new ArrayList<>();

    private boolean printEachEpoch = false;

    public AnalyticLearner(Learner wrappedLearner) {
        this.wrappedLearner = Objects.requireNonNull(wrappedLearner);
    }

    @Override
    public Vector learn(EpochInfo epochInfo) {
        Vector adjustments = wrappedLearner.learn(epochInfo);
        var previousEpoch = epochs.size() == 0 ? null : epochs.get(epochs.size() - 1);
        Epoch epoch = new Epoch(epochInfo.deepCopy(), adjustments.copy(), previousEpoch);

        if (printEachEpoch) {
            print(epoch);
        }

        epochs.add(epoch);

        return adjustments;
    }

    public void setPrintEachEpoch(boolean value) { this.printEachEpoch = value; }

    private void print(Epoch epoch) {
        System.out.println(String.format(
                "%f %f %f %f %f",
                epoch.epochInfo.error(),
                epoch.adjustments.length(),
                epoch.decomposition.parallel,
                epoch.decomposition.transverse,
                epoch.cumulativeAdjustments.length()));
    }

    public static class Epoch {
        private final EpochInfo epochInfo;
        private final Vector adjustments;
        private final Vector cumulativeAdjustments;
        private final ScalarVectorDecomposition decomposition;

        private Epoch(EpochInfo epochInfo, Vector adjustments, Epoch previousEpoch) {
            this.epochInfo = epochInfo;
            this.adjustments = adjustments;

            if (previousEpoch == null) {
                this.cumulativeAdjustments = adjustments.copy();
                this.decomposition = new ScalarVectorDecomposition(0, adjustments.length());
            } else {
                this.cumulativeAdjustments = previousEpoch.cumulativeAdjustments.copy();
                this.cumulativeAdjustments.add(adjustments);
                this.decomposition = decompose(previousEpoch.adjustments, adjustments);
            }
        }

        public EpochInfo epochInfo() { return epochInfo; }
        public Vector adjustments() { return adjustments; }
        public ScalarVectorDecomposition svDecomposition() { return decomposition; }

        private static ScalarVectorDecomposition decompose(Vector previousAdjustments, Vector adjustments) {
            // There is only one problem: If the previous epoch had a zero-vector adjustment, as it has no direction.

            Vector previousAdjustmentUnitVectorPossiblyZero = previousAdjustments.directionOrZero();
            float parallel = adjustments.dot(previousAdjustmentUnitVectorPossiblyZero);

            Vector parallelVector = adjustments.copy();
            parallelVector.multiplyScalar(parallel);

            Vector transverseVector = adjustments.copy();
            transverseVector.subtract(parallelVector);
            float transverse = transverseVector.length();

            return new ScalarVectorDecomposition(parallel, transverse);
        }
    }

    public List<Epoch> epochs() { return epochs; }

    /**
     * The adjustments vector at epoch e can be decomposed into an adjustments vector in the direction of
     * the previous adjustment vector, represented by a real number {@code parallel} - negative if it points in the opposite
     * direction of the previous adjustment vector. And a non-negative number {@code transverse} being the length
     * of the transverse vector.
     *
     * <p>SVDecomposition comes from scalar vector decomposition: Decomposition of a vector into 2 scalars
     * based on a reference/base vector.</p>
     */
    public static class ScalarVectorDecomposition {
        private final float parallel;
        private final float transverse;

        public ScalarVectorDecomposition(float parallel, float transverse) {
            this.parallel = parallel;
            this.transverse = transverse;
        }

        /** The factor to multiply the base unit vector to get the parallel vector component. */
        public float parallel() { return parallel; }

        /** The length of the transverse component. */
        public float transverse() { return transverse; }
    }
}
