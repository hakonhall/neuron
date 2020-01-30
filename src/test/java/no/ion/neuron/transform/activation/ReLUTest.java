package no.ion.neuron.transform.activation;

import no.ion.neuron.NeuralNet;
import no.ion.neuron.tensor.Vector;
import no.ion.neuron.transform.ActivationTransform;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ReLUTest {
    @Test
    void convergence() {
        var net = new NeuralNet(2);
        ActivationTransform transform = new ActivationTransform(2, new ReLU(0.1f));
        net.addTransform(transform);
        Vector output = net.compute(Vector.from(2, -3), Vector.from(4, 5));
        assertEquals(Vector.from(2, -0.3f), output);
    }
}