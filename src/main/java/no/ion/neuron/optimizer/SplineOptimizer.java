package no.ion.neuron.optimizer;

import no.ion.neuron.tensor.ScalarVectorDecomposition;
import no.ion.neuron.tensor.Vector;

public class SplineOptimizer implements Optimizer {
    private final float initialLearningRate;
    private final float growthFactor;

    private float learningRate;

    private Vector previousAdjustments = null;
    private Vector previousGradient = null;
    private float previousError = Float.NaN;

    public SplineOptimizer(float initialLearningRate, float growthFactor) {
        this.initialLearningRate = initialLearningRate;
        this.learningRate = initialLearningRate;
        this.growthFactor = growthFactor;
    }

    @Override
    public Vector calculateParameterAdjustments(EpochSummary epochSummary) {
        Vector adjustments;

        if (previousAdjustments != null && epochSummary.errorSum() > previousError) {
            float t;

            float y1 = previousError;
            float y2 = epochSummary.errorSum();
            float k1 = - previousGradient.length();

            var decomposition = ScalarVectorDecomposition.from(previousGradient, epochSummary.gradientSum());
            float k2 = - decomposition.parallel();

            float Dy = y2 - y1;
            float Dx = previousAdjustments.length();
            float K = Dy / Dx;

            float A = k1 + k2 - 2 * K;
            if (A == 0) {
                t = k1 / (k1 - k2);
            } else {
                float B = 2 * k1 + k2 - 3 * K;
                float C = B / (3 * A);
                float l = (float) Math.sqrt(C * C - k1 / (3 * A));
                float tminus = C - l;
                if (tminus > 0) {
                    t = tminus;
                } else {
                    t = C + l;
                }
            }

            if (t <= 0 || t >= 1) {
                throw new IllegalStateException("Failed to find minimum");
            }

            // result: Instead of having used previousAdjustments, we should have used t * previousAdjustments.
            // Therefore, now backtrack to that place.

            adjustments = previousAdjustments.copy();
            adjustments.multiplyScalar(t - 1);

            previousAdjustments.multiplyScalar(t);
            learningRate *= t;

            //System.out.println("learning rate = " + learningRate + ", E = " + epochSummary.errorSum() + ", t = " + t);
        } else {
            adjustments = epochSummary.gradientSum().copy();
            adjustments.multiplyScalar(- learningRate);

            previousAdjustments = adjustments.copy();
            previousGradient = epochSummary.gradientSum().copy();
            previousError = epochSummary.errorSum();
            learningRate *= growthFactor;

            //System.out.println("learning rate = " + learningRate + ", E = " + epochSummary.errorSum());
        }

        return adjustments;
    }
}
