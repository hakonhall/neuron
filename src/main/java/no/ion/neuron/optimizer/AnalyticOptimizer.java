package no.ion.neuron.optimizer;

import no.ion.neuron.tensor.ScalarVectorDecomposition;
import no.ion.neuron.tensor.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Wraps a learner and exposes various metrics that can be used to analyse the wrapped learner.
 */
public class AnalyticOptimizer implements Optimizer {
    private final Optimizer wrappedOptimizer;
    private final List<Epoch> epochs = new ArrayList<>();

    private boolean printEachEpoch = false;

    public AnalyticOptimizer(Optimizer wrappedOptimizer) {
        this.wrappedOptimizer = Objects.requireNonNull(wrappedOptimizer);
    }

    @Override
    public Vector calculateParameterAdjustments(EpochSummary epochSummary) {
        Vector adjustments = wrappedOptimizer.calculateParameterAdjustments(epochSummary);
        var previousEpoch = epochs.size() == 0 ? null : epochs.get(epochs.size() - 1);
        Epoch epoch = new Epoch(epochs.size(), epochSummary.deepCopy(), adjustments.copy(), previousEpoch);

        if (printEachEpoch) {
            print(epoch);
        }

        epochs.add(epoch);

        return adjustments;
    }

    public void setPrintEachEpoch(boolean value) { this.printEachEpoch = value; }

    private void print(Epoch epoch) {
        System.out.println(epoch);
    }

    public Epoch lastEpoch() {
        if (epochs.isEmpty()) {
            throw new IllegalStateException("There are no epochs yet");
        }

        return epochs.get(epochs.size() - 1);
    }

    public static class Epoch {
        private final int id;
        private final int epochs;
        private final EpochSummary epochSummary;
        private final Vector adjustments;
        private final Vector cumulativeAdjustments;
        private final float cumulativeLengths;
        private final ScalarVectorDecomposition decomposition;
        private final float deflectionAngleDegrees;
        private final float cumulativeDeflectionAngleSquared;

        private Epoch(int id, EpochSummary epochSummary, Vector adjustments, Epoch previousEpoch) {
            this.id = id;
            this.epochSummary = epochSummary;
            this.adjustments = adjustments;

            if (previousEpoch == null) {
                this.epochs = 1;
                this.cumulativeAdjustments = adjustments.copy();
                this.cumulativeLengths = adjustments.length();
                this.decomposition = ScalarVectorDecomposition.decomposeFromZero(adjustments);
                this.deflectionAngleDegrees = decomposition.angleDegrees();
                this.cumulativeDeflectionAngleSquared = deflectionAngleDegrees * deflectionAngleDegrees;
            } else {
                this.epochs = previousEpoch.epochs + 1;
                this.cumulativeAdjustments = previousEpoch.cumulativeAdjustments.copy();
                this.cumulativeAdjustments.add(adjustments);
                this.cumulativeLengths = previousEpoch.cumulativeLengths + adjustments.length();
                this.decomposition = ScalarVectorDecomposition.from(previousEpoch.adjustments, adjustments);
                this.deflectionAngleDegrees = decomposition.angleDegrees();
                this.cumulativeDeflectionAngleSquared =
                        previousEpoch.cumulativeDeflectionAngleSquared + deflectionAngleDegrees * deflectionAngleDegrees;
            }
        }

        public EpochSummary epochSummary() { return epochSummary; }
        public Vector adjustments() { return adjustments; }
        public ScalarVectorDecomposition svDecomposition() { return decomposition; }

        private float rmsDeflectionAngleDegrees() {
            return (float) Math.sqrt(cumulativeDeflectionAngleSquared / epochs);
        }

        @Override
        public String toString() {
            return "Epoch{" +
                    id +
                    ", \u00ca=" + (epochSummary.errorSum() / epochSummary.batchSize()) +
                    ", |\u2207p|=" + epochSummary.gradientSum().length() / epochSummary.batchSize() +
                    ", |\u0394p|=" + adjustments.length() +
                    ", \u03b8=" + deflectionAngleDegrees +
                    ", rms(\u03b8)=" + rmsDeflectionAngleDegrees() +
                    //", \u0394p\u2225=" + decomposition.parallel +
                    //", |\u0394p\u27c2|=" + decomposition.transverse +
                    ", \u2211|\u0394p|=" + cumulativeLengths +
                    ", |\u2211\u0394p|=" + cumulativeAdjustments.length() +
                    '}';
        }
    }

    public List<Epoch> epochs() { return epochs; }

    @Override
    public String toString() {
        return "AnalyticOptimizer{" +
                (epochs.isEmpty() ? "" : "epochs[-1]=" + epochs.get(epochs.size() - 1).toString()) +
                '}';
    }
}
