package no.ion.neuron.transform.mapper;

import no.ion.neuron.tensor.Vector;
import no.ion.neuron.NeuralNet;
import no.ion.neuron.optimizer.MiniBatchGradientDescent;
import no.ion.neuron.transform.MapTransform;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class LeakyReLUTest {
    @Test
    void convergence() {
        var net = new NeuralNet(2, new MiniBatchGradientDescent(1, 0.1f));
        MapTransform transform = new MapTransform(2, new LeakyReLU(0.1f));
        net.addTransform(transform);
        Vector output = net.compute(Vector.from(2, -3), Vector.from(4, 5));
        assertEquals(Vector.from(2, -0.3f), output);
    }
}