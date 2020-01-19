package no.ion.neuron;

import no.ion.neuron.tensor.Matrix;
import no.ion.neuron.tensor.Vector;
import no.ion.neuron.optimizer.MiniBatchGradientDescent;
import no.ion.neuron.transform.mapper.LeakyReLU;
import no.ion.neuron.transform.mapper.Mapper;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class XorTest2 {
    private NeuralNet net;
    private MiniBatchGradientDescent optimizer;

    @Test
    void learningXor0() {
        // We'll use a neural network with 3 layers with 2-2-1 neurons.

        // We'll use batch gradient descent - batch size is 4
        optimizer = new MiniBatchGradientDescent(1, 0.1f);
        optimizer.keepErrorHistory();

        net = new NeuralNet(2, optimizer);

        Matrix weight2 = Matrix.from(2, .1f, -.1f, .2f, .3f);
        Vector bias2 = Vector.from(0.5f, -0.3f);
        Mapper mapper2 = new LeakyReLU(0.1f);
        net.addLayer(weight2, bias2, mapper2);

        Matrix weight3 = Matrix.from(1, .4f, .5f);
        Vector bias3 = Vector.from(-.2f);
        Mapper mapper3 = mapper2;
        net.addLayer(weight3, bias3, mapper3);

        run(0, 1, 1);
    }

    @Test
    void learningXor() {
        // We'll use a neural network with 3 layers with 2-2-1 neurons.

        // We'll use batch gradient descent - batch size is 4
        optimizer = new MiniBatchGradientDescent(4, 0.1f);
        optimizer.keepErrorHistory();

        net = new NeuralNet(2, optimizer);

        Matrix weight2 = Matrix.from(2, .1f, -.1f, .2f, .3f);
        Vector bias2 = Vector.from(0.5f, -0.3f);
        Mapper mapper2 = new LeakyReLU(0.1f);
        net.addLayer(weight2, bias2, mapper2);

        Matrix weight3 = Matrix.from(1, .4f, .5f);
        Vector bias3 = Vector.from(-.2f);
        Mapper mapper3 = mapper2;
        net.addLayer(weight3, bias3, mapper3);

        runEpochs(200);
    }

    private void runEpochs(int times) {
        IntStream.range(0, times).forEach(i -> System.out.println(runEpoch()));
    }

    /** Returns the average of the square of the error across the epoch. */
    private float runEpoch() {
        int sizeBefore = optimizer.errorHistory().size();

        run(0, 0, 0);
        run(0, 1, 1);
        run(1, 0, 1);
        run(1, 1, 0);

        List<Float> errorSquaredList = optimizer.errorHistory();

        float sum = 0f;
        int count = 0;
        for (int i = sizeBefore; i < errorSquaredList.size(); ++i, ++count) {
            sum += errorSquaredList.get(i);
        }
        return sum / count;
    }

    private void run(int input1, int input2, int correctOutput1) {
        var input = Vector.from(input1, input2);
        var correctOutput = Vector.from(correctOutput1);
        net.compute(input, correctOutput);
    }

    private float lastRunErrorSquared() {
        return optimizer.errorHistory().get(optimizer.errorHistory().size() - 1);
    }
}
