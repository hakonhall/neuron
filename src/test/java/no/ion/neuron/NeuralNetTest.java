package no.ion.neuron;

import no.ion.neuron.tensor.Matrix;
import no.ion.neuron.tensor.Vector;
import no.ion.neuron.optimizer.MiniBatchGradientDescent;
import no.ion.neuron.transform.mapper.LeakyReLU;
import no.ion.neuron.transform.mapper.Mapper;
import org.junit.jupiter.api.Test;

import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;

class NeuralNetTest {
    private MiniBatchGradientDescent optimizer;
    private NeuralNet net;

    @Test
    void trivial() {
        optimizer = new MiniBatchGradientDescent(1, 0.1f);
        net = new NeuralNet(1, optimizer);
        Vector output = net.compute(Vector.from(1f), Vector.from(3f));
        assertEquals(Vector.from(1f), output);
    }

    @Test
    void trivialTwoLayers() {
        optimizer = new MiniBatchGradientDescent(1, 0.1f);

        net = new NeuralNet(1, optimizer);

        Matrix weight2 = Matrix.from(1, .1f);
        Vector bias2 = Vector.from(0.5f);
        Mapper mapper2 = new LeakyReLU(0.1f);
        net.addLayer(weight2, bias2, mapper2);

        epoch(100);
    }

    private void epoch(int count) { IntStream.range(0, count).forEach(ignored -> epoch());}

    private void epoch() {
        run(0, 0);
        run(1, 1);
    }

    @Test
    void twoLayers() {
        var optimizer = new MiniBatchGradientDescent(1, 0.1f);

        var net = new NeuralNet(2, optimizer);

        Matrix weight2 = Matrix.from(2, .1f, -.1f, .2f, .3f);
        Vector bias2 = Vector.from(0.5f, -0.3f);
        Mapper mapper2 = new LeakyReLU(0.1f);
        net.addLayer(weight2, bias2, mapper2);

        Matrix weight3 = Matrix.from(1, .4f, .5f);
        Vector bias3 = Vector.from(-.2f);
        Mapper mapper3 = mapper2;
        net.addLayer(weight3, bias3, mapper3);
    }

    private void run(int input1, int correctOutput1) {
        var input = Vector.from(input1);
        var correctOutput = Vector.from(correctOutput1);
        net.compute(input, correctOutput);
        System.out.println(net);
    }
}