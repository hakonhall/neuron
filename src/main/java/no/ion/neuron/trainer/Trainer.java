package no.ion.neuron.trainer;

import no.ion.neuron.tensor.Vector;
import no.ion.neuron.NeuralNet;

import java.util.ArrayList;
import java.util.List;

public class Trainer {
    private final List<RunInfo> runInfos = new ArrayList<>();
    private final NeuralNet net;

    public static class RunInfo {
        private final Vector input;
        private final Vector idealOutput;

        public RunInfo(Vector input, Vector idealOutput) {
            this.input = input;
            this.idealOutput = idealOutput;
        }
    }

    public Trainer(NeuralNet net) {
        this.net = net;
    }

    public Trainer add(Vector input, Vector idealOutput) {
        runInfos.add(new RunInfo(input.copy(), idealOutput.copy()));
        return this;
    }

    public void runEpoch() {
        for (var runInfo : runInfos) {
            Vector output = net.compute(runInfo.input, runInfo.idealOutput);
        }
    }

    public void runEpochs(int times) {
        while (times --> 0) runEpoch();
    }

    public void runUntilErrorIsBelow(float epsilon) {
        float weightedLastError = -1;
    }
}
