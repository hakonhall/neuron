package no.ion.neuron.trainer;

import no.ion.neuron.ComputeContext;
import no.ion.neuron.FeedForwardNeuralNet;
import no.ion.neuron.learner.Optimizer;
import no.ion.neuron.tensor.Vector;
import no.ion.neuron.transform.ErrorTransform;
import no.ion.neuron.transform.loss.ErrorFunction;

/**
 * Takes ownership of a {@link FeedForwardNeuralNet neural net}, prepares it for training,
 * and exposes method to process input/idealOutputs pairs, and a method to learn from
 * such experience with an optimizer.
 */
public class Trainer {
    private final FeedForwardNeuralNet net;
    private final Optimizer optimizer;
    private final int outputSize;

    private int processed = 0;
    private float lastError = 0;
    private float sumError = 0;

    public Trainer(FeedForwardNeuralNet net, ErrorFunction errorFunction, Optimizer optimizer) {
        this.net = net;
        this.optimizer = optimizer;
        this.outputSize = net.outputSize();

        net.addTransform(new ErrorTransform(outputSize, errorFunction));
    }

    public int outputSizeOfOriginalNet() { return outputSize; }

    public void process(Vector input, Vector idealOutput) {
        ComputeContext context = new ComputeContext(idealOutput);
        Vector error = net.compute(context, input);

        // Training a neural network means adding an artificial last layer, that transforms
        // the output vector to a single float - by the error (or loss) function. If this fails,
        // there is something wrong with ErrorTransform.
        if (error.size() != 1) {
            throw new IllegalStateException("Output of neural network did not have expected size 1: " + error.size());
        }

        ++processed;
        lastError = error.get(0);
        sumError += error.get(0);
    }

    public float averageError() {
        return sumError / processed;
    }

    public void learn() {
        if (processed <= 0) {
            // nothing to learn
            return;
        }

        Vector gradientOfParameters = net.gradientOfParameters();
        Optimizer.EpochInfo epochInfo = new Optimizer.EpochInfo(sumError, processed, gradientOfParameters);
        Vector delta = optimizer.calculateParameterAdjustments(epochInfo);
        net.adjustParameters(delta);
        net.clearCumulativeErrorGradientOfParameters();
        processed = 0;
        sumError = 0;
    }
}
