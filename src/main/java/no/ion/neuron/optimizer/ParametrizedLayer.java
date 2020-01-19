package no.ion.neuron.optimizer;

import no.ion.neuron.tensor.Vector;

public interface ParametrizedLayer {
    LayerId layerId();
    int parameterSize();
    void adjustParameters(Vector amount);
}
