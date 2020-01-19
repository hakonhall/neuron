package no.ion.neuron.transform;

import no.ion.neuron.tensor.Matrix;
import org.junit.jupiter.api.Test;

class WeightTransformTest {
    @Test
    void verifyWeightTransform() {
        var weight = Matrix.from(2,
                1, 4, -1,
                3, 1, -2);
        var transform = new WeightTransform(weight);
        var simpleLearner = new TestTrainer(3, transform);
        simpleLearner.setEpoch(
                Matrix.from(4,
                        1, 0, 0,
                        0, 1, 0,
                        0, 0, 1,
                        1, 2, 3),
                Matrix.from(4,
                        2, 1,
                        -1, -1,
                        0, 3,
                        0, 8));

        simpleLearner.runEpochsUntil(
                () -> transform.weight().toVector(),
                Matrix.from(2,
                        2, -1, 0,
                        1, -1, 3).toVector(),
                1e-5f);
    }
}