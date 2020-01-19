package no.ion.neuron.optimizer;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

public class ComputationId implements Comparable<ComputationId> {
    private static final AtomicInteger counter = new AtomicInteger(0);

    private final int id;

    public static ComputationId createNext() { return new ComputationId(); }

    private ComputationId() {
        this.id = counter.getAndIncrement();
    }

    public int id() {
        return id;
    }

    @Override
    public String toString() {
        return String.valueOf(id);
    }

    @Override
    public int compareTo(ComputationId rhs) {
        return Integer.compare(id, rhs.id);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ComputationId that = (ComputationId) o;
        return id == that.id;
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
}
