import net.jcip.annotations.Immutable;

import java.util.Arrays;

import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

/**
 * Represents a point in Rn
 */
@Immutable
public class Point {
    /**
     * Coordinates of the points in Rn space
     */
    private final float[] coordinates;

    /**
     * Constructor
     * @param coordinates coordinates of the points in Rn space
     * @throws NullPointerException if coordinates is null
     */
    public Point(final float[] coordinates) throws NullPointerException {
        if (coordinates == null) {
            throw new NullPointerException("Coordinates can't be null");
        }
        this.coordinates = Arrays.copyOf(coordinates, coordinates.length);
    }

    /**
     * Copy constructor
     * @param point point to copy
     * @throws NullPointerException if point is null
     */
    public Point(final Point point) throws NullPointerException {
        if (point == null) {
            throw new NullPointerException("Point can't be null");
        }
        this.coordinates = point.getCoordinates();
    }

    /**
     * Get the dimension of the point
     * @return dimension
     */
    public final int getDimension() {
        return coordinates.length;
    }

    /**
     * Get the i-th coordinate of the point (1-indexed)
     * @param i coordinate to get
     * @return value of i-th coordinate
     * @throws IndexOutOfBoundsException if the coordinate is <1 or >dimension
     */
    public final float getCoordinate(final int i) throws IndexOutOfBoundsException {
        if (i > coordinates.length || i < 1) {
            throw new IndexOutOfBoundsException("Invalid dimension");
        }

        return this.coordinates[i-1];
    }

    /**
     * Get all the coordinates of the point
     * @return coordinates
     */
    public final float[] getCoordinates() {
        return Arrays.copyOf(coordinates, coordinates.length);
    }

    /**
     * Calculate the euclidean distance between two points
     * @param p1 first point
     * @param p2 second point
     * @return distance
     * @throws IllegalArgumentException if the points has incompatible dimensions
     * @throws NullPointerException if one or both the points are null
     */
    public static float getEuclideanDistance(Point p1, Point p2) throws IllegalArgumentException, NullPointerException {
        return (float) sqrt(getSquaredEuclideanDistance(p1, p2));
    }

    /**
     * Calculate the squared euclidean distance between two points
     * @param p1 first point
     * @param p2 second point
     * @return distance
     * @throws IllegalArgumentException if the points has incompatible dimensions
     * @throws NullPointerException if one or both the points are null
     */
    public static float getSquaredEuclideanDistance(Point p1, Point p2) throws IllegalArgumentException, NullPointerException {
        if (p1 == null || p2 == null) {
            throw new NullPointerException("Points can't be nulls");
        }

        if (p1.getDimension() != p2.getDimension()) {
            throw new IllegalArgumentException("Incompatible point (due to its dimension)");
        }

        float squareSum = 0;
        for (int i = 0; i < p1.getDimension(); i++) {
            float difference = p1.getCoordinate(i+1) - p2.getCoordinate(i+1);
            squareSum += difference*difference;
        }
        return squareSum;
    }
}
