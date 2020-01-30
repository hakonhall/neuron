package no.ion.neuron.layer;

import no.ion.neuron.tensor.Vector;

public interface ParametrizedLayer {
    LayerId layerId();
    default int parameterSize() { return cumulativeErrorGradientOfParameters().size(); }
    Vector parameters();
    Vector cumulativeErrorGradientOfParameters();
    void clearCumulativeErrorGradientOfParameters();
    void adjustParameters(Vector amount);
}
