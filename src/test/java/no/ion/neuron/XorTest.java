package no.ion.neuron;

import no.ion.neuron.learner.FixedRateOptimizer;
import no.ion.neuron.tensor.Matrix;
import no.ion.neuron.tensor.Vector;
import no.ion.neuron.learner.AnalyticOptimizer;
import no.ion.neuron.trainer.DirectMiniBatch;
import no.ion.neuron.trainer.Trainer;
import no.ion.neuron.transform.loss.HalfErrorSquared;
import no.ion.neuron.transform.mapper.LeakyReLU;
import org.junit.jupiter.api.Test;

public class XorTest {
    @Test
    void train() {
        var net = new FeedForwardNeuralNet(2);
        // The following converges to E=0.5, which can be understood by the following problem:
        //   - a ReLU-type neural network may converge to a local minimum as-if it is
        //     the whole network is linear, and that it cannot get out of.
        //   - therefore, ensure enough logits are negative.
        //net.addLayer(Matrix.from(2, .1f, .2f, -.1f, .3f), Vector.from(.1f, .7f), new LeakyReLU(.1f));
        //net.addLayer(Matrix.from(1, .5f, -.2f), Vector.from(.4f), new LeakyReLU(.1f));

        net.addLayer(Matrix.from(2, .1f, -.1f, .2f, .3f), Vector.from(.5f, -.3f), new LeakyReLU(.1f));
        net.addLayer(Matrix.from(1, .4f, .5f), Vector.from(-.2f), new LeakyReLU(.1f));

        var optimizer = new AnalyticOptimizer(new FixedRateOptimizer(.01f));
        optimizer.setPrintEachEpoch(true);
        var trainer = new Trainer(net, new HalfErrorSquared(), optimizer);
        var miniBatch = new DirectMiniBatch(trainer);


        miniBatch
                .add(Vector.from(0, 0), Vector.from(0))
                .add(Vector.from(0, 1), Vector.from(1))
                .add(Vector.from(1, 0), Vector.from(1))
                .add(Vector.from(1, 1), Vector.from(0));

        miniBatch.runEpochs(1000);
        System.out.println(net);

        var epochs = optimizer.epochs();
    }
}
