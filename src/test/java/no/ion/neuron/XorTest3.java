package no.ion.neuron;

import no.ion.neuron.tensor.Matrix;
import no.ion.neuron.tensor.Vector;
import no.ion.neuron.learner.AnalyticLearner;
import no.ion.neuron.learner.FixedRateLearner;
import no.ion.neuron.optimizer.MiniBatchGradientDescent;
import no.ion.neuron.trainer.Trainer;
import no.ion.neuron.transform.mapper.LeakyReLU;
import org.junit.jupiter.api.Test;

public class XorTest3 {
    @Test
    void train() {
        var learner = new AnalyticLearner(new FixedRateLearner(.01f));
        learner.setPrintEachEpoch(true);
        var optimizer = new MiniBatchGradientDescent(4, learner);
        var net = new NeuralNet(2, optimizer);


        // The following converges to E=0.5, which can be understood by the following problem:
        //   - a ReLU-type neural network may converge to a local minimum as-if it is
        //     the whole network is linear, and that it cannot get out of.
        //   - therefore, ensure enough logits are negative.
        //net.addLayer(Matrix.from(2, .1f, .2f, -.1f, .3f), Vector.from(.1f, .7f), new LeakyReLU(.1f));
        //net.addLayer(Matrix.from(1, .5f, -.2f), Vector.from(.4f), new LeakyReLU(.1f));

        net.addLayer(Matrix.from(2, .1f, -.1f, .2f, .3f), Vector.from(.5f, -.3f), new LeakyReLU(.1f));
        net.addLayer(Matrix.from(1, .4f, .5f), Vector.from(-.2f), new LeakyReLU(.1f));

        var trainer = new Trainer(net)
                .add(Vector.from(0, 0), Vector.from(0))
                .add(Vector.from(0, 1), Vector.from(1))
                .add(Vector.from(1, 0), Vector.from(1))
                .add(Vector.from(1, 1), Vector.from(0));

        trainer.runEpochs(1000);
        System.out.println(net);

        var epochs = learner.epochs();
    }
}
