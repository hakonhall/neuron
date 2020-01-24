package no.ion.neuron.transform;

import no.ion.neuron.tensor.Matrix;
import no.ion.neuron.tensor.Vector;
import org.junit.jupiter.api.Test;

class BiasTransformTest {
    @Test
    void verifyConvergence() {
        BiasTransform transform = new BiasTransform(Vector.from(1, 3));
        TestTrainer trainer = new TestTrainer(2, transform);
        trainer.setEpoch(
                Matrix.from(3,
                        -3, 7,
                        4, 3,
                        6, 6),
                Matrix.from(3,
                        -3 + 5, 7 + 7,
                        4 + 5, 3 + 7,
                        6 + 5, 6 + 7));

        trainer.runEpochsUntil(transform::bias, Vector.from(5, 7), 1e-5f);
    }
}