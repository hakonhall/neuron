package no.ion.neuron.tensor;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.Supplier;

public class Matrix {
    /**
     *  The value of the matrix element at row r and column c is {@code M(r, c) = values[r * columns + c]}.
     *
     * <pre>
     *   / values[        0 * columns + 0]  values[       0 * columns + 1] ... values[       0 * columns + (columns-1)] \
     *  |  values[        1 * columns + 0]  values[       1 * columns + 1] ... values[       1 * columns + (columns-1)]  |
     *  |  values[        2 * columns + 0]  values[       2 * columns + 1] ... values[       2 * columns + (columns-1)]  |
     *  |     ...                              ...                         ...    ...                                    |
     *   \ values[(rows-1) * columns + 0]  values[(rows-1) * columns + 1]  ... values[(rows-1) * columns + (columns-1)] /
     * </pre>
     */
    private float values[];
    private final int rows;
    private final int columns;

    public static Matrix from(int numberOfRows, float... values) {
        return new Matrix(numberOfRows, values);
    }

    public Matrix(int columns, int rows, Supplier<Float> elementSupplier) {
        this(rows, columns);

        for (int i = 0; i < values.length; ++i) {
            values[i] = elementSupplier.get();
        }
    }

    public Matrix(int rows, int columns, float value) {
        this(rows, columns);
        Arrays.fill(values, value);
    }

    public Matrix(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
        this.values = new float[rows * columns];
    }

    Matrix(int rows, float... values) {
        if (values.length % rows != 0) {
            throw new IllegalArgumentException("outputSize " + rows +
                    " is not a factor of the number of values " + values.length);
        }

        if (rows <= 0) {
            throw new IllegalArgumentException("outputSize must be positive, but was: " + rows);
        }

        this.rows = rows;
        this.columns = values.length / rows;
        this.values = values;
    }

    public int rows() { return rows; }
    public int columns() { return columns; }
    public Vector dimensions() { return Vector.from(rows, columns); }

    /** Transfer the values ownership to the returned vector. Inverse of Vector.toMatrix(). */
    public Vector toVector() { return new Vector(values); }

    /** Retrieving a row vector is faster than retrieving a column vector. */
    public Vector row(int rowIndex) {
        float[] rowArray = Arrays.copyOfRange(values, rowIndex * columns, (rowIndex + 1) * columns);
        return new Vector(rowArray);
    }

    public Vector column(int columnIndex) {
        Vector column = new Vector(rows);
        for (int i = 0; i < rows; ++i) {
            column.setElement(i, indexOf(i, columnIndex));
        }
        return column;
    }

    private int indexOf(int row, int column) {
        return row * columns + column;
    }

    public float getElement(int outputIndex, int inputIndex) {
        return values[indexOf(outputIndex, inputIndex)];
    }

    /** Set the elements of the matrix. Order of elements are (outputIndex, inputIndex) sequence (0,0), (0, 1), etc */
    public void set(float... elements) {
        if (elements.length != values.length) {
            throw new IllegalArgumentException("Matrix has " + values.length + " elements, but elements array has " +
                    elements.length + " elements");
        }
        values = elements;
    }

    public void setElement(int outputIndex, int inputIndex, float value) {
        values[indexOf(outputIndex, inputIndex)] = value;
    }

    public void addToElement(int outputIndex, int inputIndex, float value) {
        values[indexOf(outputIndex, inputIndex)] += value;
    }

    public void add(Matrix rhs) {
        if (rows != rhs.rows) {
            throw new IllegalArgumentException("Output sizes differ");
        }

        if (columns != rhs.columns) {
            throw new IllegalArgumentException("Input sizes differ");
        }

        for (int i = 0; i < values.length; ++i) {
            values[i] += rhs.values[i];
        }
    }

    public Vector dot(Vector rhs) {
        if (rhs.size() != columns) {
            throw new IllegalArgumentException("Matrix of dimension " + rows + "x" + columns +
                    " cannot be multiplied with a vector of dimension " + rhs.size());
        }

        Vector result = new Vector(rows);
        for (int outputIndex = 0; outputIndex < rows; ++outputIndex) {
            float sum = 0.0f;
            for (int inputIndex = 0; inputIndex < columns; ++inputIndex) {
                sum += getElement(outputIndex, inputIndex) * rhs.get(inputIndex);
            }
            result.setElement(outputIndex, sum);
        }

        return result;
    }

    public void clear() {
        Arrays.fill(values, 0f);
    }

    @Override
    public String toString() {
        var builder = new StringBuilder();
        builder.append("[");

        for (int i = 0; i < rows; ++i) {
            if (i == 0) {
                builder.append("[");
            } else {
                builder.append(",[");
            }

            for (int j = 0; j < columns; ++j) {
                if (j > 0) {
                    builder.append(",");
                }
                builder.append(getElement(i, j));
            }
            builder.append("]");
        }

        builder.append("]");

        return builder.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Matrix matrix = (Matrix) o;
        return rows == matrix.rows &&
                Arrays.equals(values, matrix.values);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(rows);
        result = 31 * result + Arrays.hashCode(values);
        return result;
    }
}
