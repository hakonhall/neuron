package no.ion.neuron.optimizer;

import no.ion.neuron.tensor.Vector;

public interface Optimizer {

    /** Calculate the adjustment to each parameter given information on a training set. */
    Vector calculateParameterAdjustments(EpochSummary epochSummary);
}
