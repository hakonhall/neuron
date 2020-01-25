package no.ion.neuron.learner;

import no.ion.neuron.optimizer.LayerId;

import java.util.Iterator;

public interface MiniBatchGradient {

    int parameterSize();
    int layerSize();

    interface LayerParameters extends Iterable<Float> {
        LayerId layerId();
        int parameterSize();
        LayerParameters previousLayer();
        LayerParameters nextLayer();
    }

    interface LayerIterable extends Iterable<LayerParameters> {

    }
}
