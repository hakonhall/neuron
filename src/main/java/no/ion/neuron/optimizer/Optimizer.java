package no.ion.neuron.optimizer;

import no.ion.neuron.tensor.Vector;

/**
 * Optimization strategy for the neural network training for gradient descent method.
 */
public interface Optimizer {
    void startComputation(ComputationId computationId, Vector input, Vector idealOutput);

    /** Optimizer takes ownership of errorGradient. */
    void registerErrorGradientOfParameters(ComputationId computationId, ParametrizedLayer layer, Vector errorGradient);

    /** Returns the error gradient w.r.t. the output of the neural net. */
    Vector errorGradient(ComputationId computationId, Vector output);

    void endComputation(ComputationId computationId);
}
