package no.ion.neuron.optimizer;

import no.ion.neuron.tensor.Vector;

public interface ParametrizedLayer {
    LayerId layerId();
    default int parameterSize() { return cumulativeErrorGradientOfParameters().size(); }
    Vector cumulativeErrorGradientOfParameters();
    void clearCumulativeErrorGradientOfParameters();
    void adjustParameters(Vector amount);
}
