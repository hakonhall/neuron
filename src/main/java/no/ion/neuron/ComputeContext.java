package no.ion.neuron;

import no.ion.neuron.tensor.Vector;

import java.util.Objects;

public class ComputeContext {
    private final Vector idealOutput;

    public ComputeContext(Vector idealOutput) {
        this.idealOutput = Objects.requireNonNull(idealOutput);
    }

    public Vector idealOutput() { return idealOutput; }
}
