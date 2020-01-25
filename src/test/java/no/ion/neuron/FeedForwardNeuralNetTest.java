package no.ion.neuron;

import no.ion.neuron.learner.FixedRateOptimizer;
import no.ion.neuron.tensor.Matrix;
import no.ion.neuron.tensor.Vector;
import no.ion.neuron.trainer.DirectMiniBatch;
import no.ion.neuron.trainer.Trainer;
import no.ion.neuron.transform.loss.HalfErrorSquared;
import no.ion.neuron.transform.mapper.LeakyReLU;
import no.ion.neuron.transform.mapper.Mapper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class FeedForwardNeuralNetTest {
    private FeedForwardNeuralNet net;

    @Test
    void trivial() {
        net = new FeedForwardNeuralNet(1);
        Vector output = net.compute(Vector.from(1f), Vector.from(3f));
        assertEquals(Vector.from(1f), output);
    }

    @Test
    void trivialTwoLayers() {
        net = new FeedForwardNeuralNet(1);
        Trainer trainer = new Trainer(net, new HalfErrorSquared(), new FixedRateOptimizer(0.1f));
        DirectMiniBatch miniBatch = new DirectMiniBatch(trainer);

        Matrix weight2 = Matrix.from(1, .1f);
        Vector bias2 = Vector.from(0.5f);
        Mapper mapper2 = new LeakyReLU(0.1f);
        net.addLayer(weight2, bias2, mapper2);

        miniBatch
                .add(Vector.from(0), Vector.from(0))
                .add(Vector.from(1), Vector.from(1));
        miniBatch.runEpochs(100);
    }
}