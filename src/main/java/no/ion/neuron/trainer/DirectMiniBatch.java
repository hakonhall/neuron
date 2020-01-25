package no.ion.neuron.trainer;

import no.ion.neuron.FeedForwardNeuralNet;
import no.ion.neuron.learner.Optimizer;
import no.ion.neuron.tensor.Vector;
import no.ion.neuron.transform.loss.ErrorFunction;

import java.util.ArrayList;
import java.util.List;

/**
 * A trainer that can be set up with a list of input-idealOutput pairs,
 * and run those as an epoch, completing each such epoch with learning.
 */
public class DirectMiniBatch {
    private final List<RunInfo> trainingSet = new ArrayList<>();
    private final Trainer trainer;

    public DirectMiniBatch(Trainer trainer) {
        this.trainer = trainer;
    }

    public DirectMiniBatch add(Vector input, Vector idealOutput) {
        trainingSet.add(new RunInfo(input.copy(), idealOutput.copy()));
        return this;
    }

    public float runEpoch() {
        for (var runInfo : trainingSet) {
            trainer.process(runInfo.input, runInfo.idealOutput);
        }

        float E = trainer.averageError();

        trainer.learn();

        return E;
    }

    public void runEpochs(int times) {
        while (times --> 0) runEpoch();
    }

    public int runUntilAverageErrorInEpochIsBelow(float epsilon, int maxEpochs) {
        int i = 0;
        for (; i < maxEpochs && runEpoch() > epsilon; ++i)
            ;

        return i;
    }

    private static class RunInfo {
        private final Vector input;
        private final Vector idealOutput;

        public RunInfo(Vector input, Vector idealOutput) {
            this.input = input;
            this.idealOutput = idealOutput;
        }
    }
}
