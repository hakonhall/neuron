package no.ion.neuron;

import no.ion.neuron.optimizer.AnalyticOptimizer;
import no.ion.neuron.optimizer.FixedRateOptimizer;
import no.ion.neuron.optimizer.SplineOptimizer;
import no.ion.neuron.tensor.Matrix;
import no.ion.neuron.tensor.Vector;
import no.ion.neuron.trainer.DirectMiniBatch;
import no.ion.neuron.trainer.Trainer;
import no.ion.neuron.transform.activation.ReLU;
import no.ion.neuron.transform.activation.Tanh;
import no.ion.neuron.transform.loss.HalfErrorSquared;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class XorTest {
    private static final boolean PRINT_DEBUG = false;

    @Test
    void simpleTrain() {
        var net = new NeuralNet(2);
        // The following converges to E=0.5, which can be understood by the following problem:
        //   - a ReLU-type neural network may converge to a local minimum as-if it is
        //     the whole network is linear, and that it cannot get out of.
        //   - therefore, ensure enough logits are negative.
        //net.addLayer(Matrix.from(2, .1f, .2f, -.1f, .3f), Vector.from(.1f, .7f), new ReLU(.1f));
        //net.addLayer(Matrix.from(1, .5f, -.2f), Vector.from(.4f), new ReLU(.1f));

        net.addLayers(Matrix.from(2, .1f, -.1f, .2f, .3f), Vector.from(.5f, -.3f), new ReLU(.1f));
        net.addLayers(Matrix.from(1, .4f, .5f), Vector.from(-.2f), new ReLU(.1f));

        var optimizer = new AnalyticOptimizer(new FixedRateOptimizer(.01f));
        optimizer.setPrintEachEpoch(PRINT_DEBUG);
        var trainer = new Trainer(net, new HalfErrorSquared(), optimizer);
        var miniBatch = new DirectMiniBatch(trainer);


        miniBatch
                .add(Vector.from(0, 0), Vector.from(0))
                .add(Vector.from(0, 1), Vector.from(1))
                .add(Vector.from(1, 0), Vector.from(1))
                .add(Vector.from(1, 1), Vector.from(0));

        int i = miniBatch.runUntilAverageErrorInEpochIsBelow(0.001f, 10000);
        if (i == 10000) {
            throw new IllegalStateException("Ran too many times: " + i);
        }

        assertEquals(1192, i);

        if (PRINT_DEBUG) {
            System.out.println("Completed after " + i + " epochs");
            System.out.println(net);
        }
    }

    @Test
    void tanhWithScaling() {
        // Using 0.08f will make the net converge towards outputting 0.5 for every input,
        // which is a local minimum when all activation function f(z) always gets z
        // with the same sign throughout the epoch. It can then be shown that the net
        // is identical to a reduced net in which the minimum is reached when the output
        // is 0.5.
        //net.addLayers(Matrix.from(2, .1f, -.1f, .2f, .3f), Vector.from(.5f, -.3f), new ReLU(epsilon));
        //net.addLayers(Matrix.from(1, .4f, .5f), Vector.from(-.2f), new ReLU(epsilon));
        //runWith(0.08f); <-- epsilon on ReLU

        runWith(.00001f);
        runWith(.02f);
        runWith(.04f); // lowest with FixedRateOptimizer
        runWith(.06f);
        runWith(.08f);
        runWith(.10f);
        runWith(.12f);
        runWith(.14f);
        runWith(.16f);
        runWith(.18f);
        runWith(.20f);
        runWith(.22f);
        runWith(.24f);
        runWith(.24f);
        runWith(.25f);
        runWith(.26f);
        runWith(.28f);
        runWith(.30f);
        runWith(.32f);
        runWith(.34f);
        runWith(.36f);
        runWith(.38f);
        runWith(.40f);
        runWith(.42f);
        runWith(.44f);
        runWith(.46f);
        runWith(.48f);
        runWith(.50f);
        runWith(.52f); // highest with FixedRateOptimizer
        runWith(.60f);
        runWith(2.50f);

        //runWith(.55f);
        //runWith(.6f);
    }

    @Test
    void tanhWithScaling2() {
        runWith(2.5f);
    }

    private AnalyticOptimizer.Epoch runWith(float learningRate) {
        var net = new NeuralNet(2);
        // The following converges to E=0.5, which can be understood by the following problem:
        //   - a ReLU-type neural network may converge to a local minimum as-if it is
        //     the whole network is linear, and that it cannot get out of.
        //   - therefore, ensure enough logits are negative.
        //net.addLayer(Matrix.from(2, .1f, .2f, -.1f, .3f), Vector.from(.1f, .7f), new ReLU(.1f));
        //net.addLayer(Matrix.from(1, .5f, -.2f), Vector.from(.4f), new ReLU(.1f));

        var activationFunction = new Tanh();
        net.addLayers(Matrix.from(2, .1f, -.1f, .2f, .3f), Vector.from(.5f, -.3f), activationFunction);
        net.addLayers(Matrix.from(1, .4f, .5f), Vector.from(-.2f), activationFunction);

        var optimizer = new SplineOptimizer(learningRate, 1.05f);
        //var optimizer = new FixedRateOptimizer(learningRate);
        var analyticOptimizer = new AnalyticOptimizer(optimizer);
        analyticOptimizer.setPrintEachEpoch(PRINT_DEBUG);
        var trainer = new Trainer(net, new HalfErrorSquared(), analyticOptimizer);
        var miniBatch = new DirectMiniBatch(trainer);

        miniBatch
                .add(Vector.from(0, 0), Vector.from(0))
                .add(Vector.from(0, 1), Vector.from(1))
                .add(Vector.from(1, 0), Vector.from(1))
                .add(Vector.from(1, 1), Vector.from(0));

        final int maxEpochs = 10000;
        int i = miniBatch.runUntilAverageErrorInEpochIsBelow(0.001f, maxEpochs);
        if (i == maxEpochs) {
            System.out.println(analyticOptimizer.lastEpoch());
            System.out.println(net);
            throw new IllegalStateException("Ran too many times: " + i);
        }

        // System.out.println(String.format("%.6f: ", learningRate) + analyticOptimizer.lastEpoch());

        return null;
    }
}
