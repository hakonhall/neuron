package no.ion.neuron.transform;

import no.ion.neuron.tensor.Matrix;
import no.ion.neuron.tensor.Vector;
import no.ion.neuron.NeuralNet;
import no.ion.neuron.optimizer.MiniBatchGradientDescent;
import no.ion.neuron.transform.loss.ErrorSquared;

import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.fail;

/**
 * A class for training a single layer with 2 inputs.
 */
public class TestTrainer {
    private final MiniBatchGradientDescent optimizer;
    private final NeuralNet net;

    // epoch variables
    private Matrix correctOutputs;
    private Matrix inputs;

    public TestTrainer(int inputSize, Transform transform) {
        this(inputSize);
        net.addTransform(transform);
        net.addTransform(new OutputTransform2(net.outputSize(), new ErrorSquared()));
    }

    public TestTrainer(int inputSize) {
        optimizer = new MiniBatchGradientDescent(1, 0.1f);
        net = new NeuralNet(inputSize, optimizer);
    }

    public NeuralNet net() { return net; }

    public void run(Vector input, Vector correctOutput) {
        net.compute(input, correctOutput);
    }

    /**
     * {@code inputs} must have the same dimensions as {@code correctOutputs}. Each pair of rows
     * in {@code inputs} and {@code correctOutputs} is run in an {@link #epoch()}.
     */
    public void setEpoch(Matrix inputs, Matrix correctOutputs) {
        if (inputs.columns() != net.inputSize()) {
            throw new IllegalArgumentException("The number of columns in inputs doesn't match network input size");
        }

        this.inputs = inputs;
        this.correctOutputs = correctOutputs;
    }

    public void epoch() {
        if (inputs == null || correctOutputs == null) {
            throw new IllegalStateException("setEpoch() has not been called");
        }

        for (int rowIndex = 0; rowIndex < inputs.rows(); ++rowIndex) {
            Vector input = inputs.row(rowIndex);
            Vector correctOutput = correctOutputs.row(rowIndex);
            run(input, correctOutput);
        }

        System.out.println(net);
    }

    public void runEpochsUntil(Supplier<Vector> parameterSupplier, Vector correctParameters, float rmseLimit) {
        for (int i = 0; i < 1000; ++i) {
            epoch();

            Vector currentParameters = parameterSupplier.get();

            Vector parameterError = currentParameters.copy();
            parameterError.subtract(correctParameters);
            double rmse = Math.sqrt(parameterError.squared());
            if (rmse < rmseLimit) {
                System.out.println("Parameters converged after " + i + " iterations: " + currentParameters);
                return;
            }
        }

        fail();
    }
}
