package no.ion.neuron.optimizer;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

public class LayerId implements Comparable<LayerId> {
    private static final AtomicInteger counter = new AtomicInteger(0);
    private final int id;

    public static LayerId createNext() { return new LayerId(); }

    private LayerId() {
        this.id = counter.getAndIncrement();
    }

    @Override public String toString() { return String.valueOf(id); }

    @Override public int compareTo(LayerId rhs) { return Integer.compare(id, rhs.id); }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        LayerId layerId = (LayerId) o;
        return id == layerId.id;
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
}
