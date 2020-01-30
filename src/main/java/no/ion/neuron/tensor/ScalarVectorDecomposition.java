package no.ion.neuron.tensor;

/**
 * A scalar representation of how a vector compares to a base vector: It's length along the base
 * vector (may be negative), and a length of the transverse vector (always non-negative).
 * A vector is transverse to a zero-length base vector.
 */
public class ScalarVectorDecomposition {
    private final float parallel;
    private final float transverse;

    /** Decompose vector along a base vector: A vector parallel to base, and transverse to base. */
    public static ScalarVectorDecomposition from(Vector base, Vector vector) {
        // There is only one potential problem:
        //  - If the previous epoch had a zero-vector adjustment it has no direction.
        Vector baseUnitVector = base.directionOrZero();
        float parallel = vector.dot(baseUnitVector);

        Vector parallelVector = baseUnitVector;
        baseUnitVector = null;
        parallelVector.multiplyScalar(parallel);

        Vector transverseVector = vector.copy();
        transverseVector.subtract(parallelVector);
        float transverse = transverseVector.length();

        return new ScalarVectorDecomposition(parallel, transverse);
    }

    /** Decompose vector along the zero vector. */
    public static ScalarVectorDecomposition decomposeFromZero(Vector vector) {
        return new ScalarVectorDecomposition(0, vector.length());
    }

    public ScalarVectorDecomposition(float parallel, float transverse) {
        this.parallel = parallel;
        this.transverse = transverse;
    }

    /** The factor to multiply the base unit vector to get the parallel vector component. */
    public float parallel() { return parallel; }

    /** The length of the transverse component. */
    public float transverse() { return transverse; }

    /** The angle the vector makes with the base vector. The angle is unspecified if the length is zero. */
    public float angle() { return (float) Math.atan2(parallel, transverse); }
    public float angleDegrees() { return (float) Math.toDegrees(angle()); }
    public float rotations() { return (float) (angle() / (2 * Math.PI)); }
}
