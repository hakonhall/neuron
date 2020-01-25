package no.ion.neuron.optimizer;

import no.ion.neuron.tensor.Vector;

/**
 * Optimization strategy for the neural network training for gradient descent method.
 */
public interface GradientGatherer {
    void startComputation(ComputationId computationId, Vector input, Vector idealOutput);

    /** GradientGatherer takes ownership of errorGradient. */
    void registerErrorGradientOfParameters(ComputationId computationId, ParametrizedLayer layer, Vector errorGradient);

    void endComputation(ComputationId computationId, Vector output);
}
