package no.ion.neuron;

import no.ion.neuron.optimizer.FixedRateOptimizer;
import no.ion.neuron.tensor.Matrix;
import no.ion.neuron.tensor.Vector;
import no.ion.neuron.trainer.DirectMiniBatch;
import no.ion.neuron.trainer.Trainer;
import no.ion.neuron.transform.activation.ReLU;
import no.ion.neuron.transform.loss.HalfErrorSquared;
import no.ion.neuron.transform.activation.ActivationFunction;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class NeuralNetTest {
    private NeuralNet net;

    @Test
    void trivial() {
        net = new NeuralNet(1);
        Vector output = net.compute(Vector.from(1f), Vector.from(3f));
        assertEquals(Vector.from(1f), output);
    }

    @Test
    void trivialTwoLayers() {
        net = new NeuralNet(1);
        Trainer trainer = new Trainer(net, new HalfErrorSquared(), new FixedRateOptimizer(0.1f));
        DirectMiniBatch miniBatch = new DirectMiniBatch(trainer);

        Matrix weight2 = Matrix.from(1, .1f);
        Vector bias2 = Vector.from(0.5f);
        ActivationFunction activationFunction2 = new ReLU(0.1f);
        net.addLayers(weight2, bias2, activationFunction2);

        miniBatch
                .add(Vector.from(0), Vector.from(0))
                .add(Vector.from(1), Vector.from(1));
        miniBatch.runEpochs(100);
    }
}