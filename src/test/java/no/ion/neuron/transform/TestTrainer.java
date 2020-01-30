package no.ion.neuron.transform;

import no.ion.neuron.NeuralNet;
import no.ion.neuron.optimizer.FixedRateOptimizer;
import no.ion.neuron.tensor.Matrix;
import no.ion.neuron.tensor.Vector;
import no.ion.neuron.trainer.DirectMiniBatch;
import no.ion.neuron.trainer.Trainer;
import no.ion.neuron.transform.loss.HalfErrorSquared;

import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.fail;

/**
 * A class for training a single layer with 2 inputs.
 */
public class TestTrainer {
    private final NeuralNet net;
    private final Trainer trainer;
    private final DirectMiniBatch miniBatch;

    // epoch variables
    private Matrix correctOutputs;
    private Matrix inputs;
    private boolean printDebug;

    public TestTrainer(int inputSize, Transform transform) {
        this.net = new NeuralNet(inputSize);
        net.addTransform(transform);
        this.trainer = new Trainer(net, new HalfErrorSquared(), new FixedRateOptimizer(0.1f));
        this.miniBatch = new DirectMiniBatch(trainer);
    }

    public void setDebug(boolean printDebug) {
        this.printDebug = printDebug;
    }

    /**
     * {@code inputs} must have the same dimensions as {@code correctOutputs}. Each pair of rows
     * in {@code inputs} and {@code correctOutputs} is run in an {@link #epoch()}.
     */
    public void setEpoch(Matrix inputs, Matrix correctOutputs) {
        if (inputs.rows() != correctOutputs.rows()) {
            throw new IllegalArgumentException("Each row of inputs and correctOutputs is supposed to be one sample, " +
                    "but they are not equal: " + inputs.rows() + " and " + correctOutputs.rows());
        }

        if (inputs.columns() != net.inputSize()) {
            throw new IllegalArgumentException("The number of columns in inputs doesn't match network input size");
        }

        if (correctOutputs.columns() != trainer.outputSizeOfOriginalNet()) {
            throw new IllegalArgumentException("The number of columns in correctOutputs is supposed to be " +
                    "the size of the output, but the sizes are not equal: " + correctOutputs.columns() + " and " +
                    trainer.outputSizeOfOriginalNet());
        }

        for (int sample = 0; sample < inputs.rows(); ++sample) {
            miniBatch.add(inputs.row(sample), correctOutputs.row(sample));
        }
    }

    public void epoch() {
        miniBatch.runEpoch();

        if (printDebug) {
            System.out.println(net);
        }
    }

    public void runEpochsUntil(Supplier<Vector> parameterSupplier, Vector correctParameters, float rmseLimit) {
        for (int i = 0; i < 1000; ++i) {
            epoch();

            Vector currentParameters = parameterSupplier.get();

            Vector parameterError = currentParameters.copy();
            parameterError.subtract(correctParameters);
            double rmse = Math.sqrt(parameterError.squared());
            if (rmse < rmseLimit) {
                if (printDebug) {
                    System.out.println("Parameters converged after " + i + " iterations: " + currentParameters);
                }
                return;
            }
        }

        fail();
    }
}
