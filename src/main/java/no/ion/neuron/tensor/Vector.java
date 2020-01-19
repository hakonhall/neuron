package no.ion.neuron.tensor;

import java.util.Arrays;
import java.util.function.Supplier;

public class Vector {
    private float values[];

    public static Vector from(float... values) {
        return new Vector(values);
    }
    public static Vector create(int size, float value) {
        return new Vector(size, value);
    }

    public Vector(Vector vector) {
        this.values = vector.values.clone();
    }

    public Vector copy() {
        return new Vector(this);
    }

    public Vector(int size) {
        this.values = new float[size];
    }

    public Vector(int size, float value) {
        this.values = new float[size];
        Arrays.fill(this.values, value);
    }

    public Vector(int size, Supplier<Float> elementSupplier) {
        values = new float[size];

        for (int i = 0; i < size; ++i) {
            values[i] = elementSupplier.get();
        }
    }

    Vector(float... values) {
        this.values = values;
    }

    public int size() { return values.length; }

    public float squared() {
        float sum = 0;
        for (int i = 0; i < values.length; ++i) {
            sum += values[i] * values[i];
        }
        return sum;
    }

    /** Also known as magnitude. */
    public float length() {
        return (float) Math.sqrt(squared());
    }

    public float get(int index) {
        return values[index];
    }

    public void setElement(int index, float value) {
        values[index] = value;
    }

    public void addToElement(int index, float value) {
        values[index] += value;
    }

    public void set(float... elements) {
        if (elements.length != values.length) {
            throw new IllegalArgumentException("Vector has " + values.length + " elements, but elements argument has " +
                    elements.length);
        }

        values = elements;
    }

    public void add(Vector rhs) {
        for (int index = 0; index < values.length; ++index) {
            values[index] += rhs.get(index);
        }
    }

    public void subtract(Vector rhs) {
        for (int index = 0; index < values.length; ++index) {
            values[index] -= rhs.get(index);
        }
    }

    public void multiplyScalar(float factor) {
        for (int index = 0; index < values.length; ++index) {
            values[index] *= factor;
        }
    }

    public void scale(Vector scales) {
        if (scales.size() != size()) {
            throw new IllegalArgumentException("A vector scale operation requires a vector of the same size");
        }

        for (int index = 0; index < values.length; ++index) {
            values[index] *= scales.values[index];
        }
    }

    public float dot(Vector rhs) {
        if (rhs.size() != size()) {
            throw new IllegalArgumentException("A vector dot operation requires vectors of same sizes");
        }

        float result = 0;
        for (int i = 0; i < size(); ++i) {
            result += values[i] * rhs.values[i];
        }
        return result;
    }

    /** Returns a unit vector (of length 1) pointing in the same direction as this, or a zero vector if length is 0. */
    public Vector directionOrZero() {
        float length = length();
        if (length == 0) {
            return new Vector(size());
        } else {
            var unit = copy();
            unit.multiplyScalar(1 / length);
            return unit;
        }
    }

    public void clear() {
        Arrays.fill(values, 0f);
    }

    /** Transfer ownership of values to returned Matrix. Inverse of Matrix.toVector(). */
    public Matrix toMatrix(int outputSize) { return new Matrix(outputSize, values); }

    @Override
    public String toString() {
        return Arrays.toString(values);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Vector vector = (Vector) o;
        return Arrays.equals(values, vector.values);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(values);
    }
}
