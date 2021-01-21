import net.jcip.annotations.Immutable;

import java.util.Arrays;

/**
 * Represents a domain for points. It's an iper-rectangular subset of Rn, eventually Rn itself.
 * A point X in a domain must satisfy LB_i <= X_i <= UB_i for i = 1...n
 */
@Immutable
public final class Domain {
    /**
     * Dimension of the Rn space
     */
    private final int dimension;

    /**
     * Array of lower-bounds
     */
    private final float[] lowerBounds;

    /**
     * Array of upper-bounds
     */
    private final float[] upperBounds;


    /**
     * Constructor
     * @param dimension dimension of the space
     * @param lowerBounds array of lower-bounds
     * @param upperBounds array of upper-bounds
     * @throws NullPointerException if lower or upper bounds are null
     * @throws IllegalArgumentException if lower or upper bounds doesn't respect the domain dimension
     */
    private Domain(final int dimension, final float[] lowerBounds, final float[] upperBounds) throws NullPointerException, IllegalArgumentException {
        if (lowerBounds == null || upperBounds == null) {
            throw new NullPointerException("Lower and upper bounds can't be null");
        }
        if (lowerBounds.length != dimension || upperBounds.length != dimension) {
            throw new IllegalArgumentException("Lower and upper bounds must respect the domain size");
        }
        this.dimension = dimension;
        this.lowerBounds = Arrays.copyOf(lowerBounds, lowerBounds.length);
        this.upperBounds = Arrays.copyOf(upperBounds, upperBounds.length);
    }

    /**
     * Checks if represents the same domain of another
     * @param domain domain to compare with
     * @return true if represents the same domain
     * @throws NullPointerException if domain is null
     */
    public final boolean equals(final Domain domain) throws NullPointerException {
        if (domain == null) {
            throw new NullPointerException("Domain can't be null");
        }
        if (dimension == domain.getDimension()) {
            return Arrays.equals(lowerBounds, domain.getLowerBounds()) && Arrays.equals(upperBounds, domain.getUpperBounds());
        }
        return false;
    }

    /**
     * Checks if a given point belongs to this domain
     * @param point point to check
     * @return true if the point belongs to this domain
     * @throws NullPointerException if point is null
     */
    public final boolean contains(final Point point) throws NullPointerException {
        if (point == null) {
            throw new NullPointerException("Point can't be null");
        }
        if (point.getDimension() == dimension) {
            for (int i = 0; i < dimension; i++) {
                try {
                    float p_i = point.getCoordinate(i+1);
                    if (p_i < lowerBounds[i] || p_i > upperBounds[i])
                        return false;
                } catch (IndexOutOfBoundsException ignore) { }
            }
            return true;
        }
        return false;
    }

    /**
     * Creates the Euclidean space Rn
     * @param dimension dimensions of Rn
     * @return Rn
     * @throws IllegalArgumentException if dimension is < 0
     */
    public static Domain Rn(final int dimension) throws IllegalArgumentException {
        if (dimension < 0) {
            throw new IllegalArgumentException("Domain dimension must be non-negative");
        }
        float[] lowerUnbound = new float[dimension];
        float[] upperUnbound = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            lowerUnbound[i] = Float.NEGATIVE_INFINITY;
            upperUnbound[i] = Float.POSITIVE_INFINITY;
        }
        return new Domain(dimension, lowerUnbound, upperUnbound);
    }

    /**
     * Creates the RGB space
     * @return RGB space
     */
    public static Domain RGB() {
        float[] lowerBounds = new float[3];
        float[] upperBounds = new float[3];
        for (int i = 0; i < 3; i++) {
            lowerBounds[i] = 0;
            upperBounds[i] = 255;
        }
        return new Domain(3, lowerBounds, upperBounds);
    }

    /**
     * @return dimension of the space
     */
    public final int getDimension() {
        return dimension;
    }

    /**
     * @return array of lower-bounds
     */
    public final float[] getLowerBounds() {
        return Arrays.copyOf(lowerBounds, lowerBounds.length);
    }

    /**
     * @return array of upper-bounds
     */
    public final float[] getUpperBounds() {
        return Arrays.copyOf(upperBounds, upperBounds.length);
    }
}
