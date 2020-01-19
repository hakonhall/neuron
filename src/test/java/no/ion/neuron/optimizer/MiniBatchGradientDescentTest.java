package no.ion.neuron.optimizer;

import no.ion.neuron.tensor.Vector;
import no.ion.neuron.learner.Learner;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.atLeast;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

class MiniBatchGradientDescentTest {
    @Test
    void trivial() {
        Learner learner = mock(Learner.class);
        MiniBatchGradientDescent optimizer = new MiniBatchGradientDescent(1, learner);
        optimizer.keepErrorHistory();

        Vector input1 = Vector.from(1, 2);

        // Model
        //   y0 = a x0
        //   y1 = b x1 + c
        //
        // Ideal parameters
        //   y0 =  -x0
        //   y1 = 2 x1 - 1
        //
        //   1 -> -1
        //   2 -> 3
        Vector idealOutput1 = Vector.from(-1, 3);

        // Initial parameters
        //   y0 = 2 x0
        //   y1 =  -x1 + 3
        //
        //   1 -> 2
        //   2 -> 1
        Vector output1 = Vector.from(2, 1);

        ComputationId computationId1 = ComputationId.createNext();
        optimizer.startComputation(computationId1, input1, idealOutput1);
        Vector outputGradient = optimizer.errorGradient(computationId1, output1);

        Vector error1 = output1.copy();
        error1.subtract(idealOutput1);
        assertEquals(Vector.from(3, -2), error1);
        assertEquals(error1, outputGradient);
        assertEquals(1, optimizer.errorHistory().size());
        float E = (3 * 3 + 2 * 2) / 2f;
        assertEquals(E, error1.squared() / 2f);
        assertEquals(E, optimizer.errorHistory().get(0));

        ParametrizedLayer layer = mock(ParametrizedLayer.class);
        LayerId layerId = LayerId.createNext();
        when(layer.layerId()).thenReturn(layerId);
        when(layer.parameterSize()).thenReturn(3);

        // Parameters:
        //   p = [a b c]
        //   p0 = [2 -1 3]
        //   pideal = [-1 2 -1]

        Vector errorGradient = Vector.from(
                3 * (-1),
                -2 * 2,
                -2);
        optimizer.registerErrorGradientOfParameters(computationId1, layer, errorGradient);

        verify(layer, times(1)).parameterSize();
        verify(layer, atLeast(0)).layerId();
        verifyNoMoreInteractions(layer);

        // This makes learner the same as FixedRateLearner(0.1).
        when(learner.learn(any())).thenReturn(Vector.from(0.3f, 0.4f, 0.2f));

        optimizer.endComputation(computationId1);

        var epochInfoCaptor = ArgumentCaptor.forClass(Learner.EpochInfo.class);
        var ECaptor = ArgumentCaptor.forClass(Float.class);
        var batchSizeCaptor = ArgumentCaptor.forClass(Integer.class);
        var gradientCaptor = ArgumentCaptor.forClass(Vector.class);
        verify(learner, times(1)).learn(epochInfoCaptor.capture());
        assertEquals(E, epochInfoCaptor.getValue().error());
        assertEquals(1, epochInfoCaptor.getValue().batchSize());
        assertEquals(errorGradient, epochInfoCaptor.getValue().gradient());

        ArgumentCaptor<Vector> deltaCaptor = ArgumentCaptor.forClass(Vector.class);
        verify(layer, times(1)).adjustParameters(deltaCaptor.capture());
        Vector delta = deltaCaptor.getValue();
        assertEquals(Vector.from(0.3f, 0.4f, 0.2f), delta);

        verifyNoMoreInteractions(layer);
    }
}