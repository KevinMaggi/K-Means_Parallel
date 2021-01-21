import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collection;

/**
 * Represents a set of points all belonging the same domain
 */
public class SetOfPoints<T extends Point> {
    /**
     * Collected points
     */
    protected final ArrayList<T> points;
    /**
     * Common domain of the points
     */
    private final Domain domain;

    /**
     * Constructor
     * @param domain domain to which the points belong
     * @throws NullPointerException if the domain is null
     */
    public SetOfPoints(final Domain domain) throws NullPointerException {
        if (domain == null) {
            throw new NullPointerException("Domain can't be null");
        }
        this.points = new ArrayList<>();
        this.domain = domain;
    }

    /**
     * Constructor
     * @param domain domain to which the points belong
     * @param points points to collect
     * @throws NullPointerException if the domain is null
     */
    public SetOfPoints(final Domain domain, final Collection<T> points) throws NullPointerException {
        if (domain == null) {
            throw new NullPointerException("Domain can't be null");
        }
        this.domain = domain;
        if (points != null) {
            this.points = new ArrayList<>(points);
        } else {
            this.points = new ArrayList<>();
        }
    }

    /**
     * Copy constructor
     * @param sop Set Of Points to copy
     * @throws NullPointerException if the set of point to copy is null
     */
    public SetOfPoints(final SetOfPoints<T> sop) throws NullPointerException {
        if (sop == null) {
            throw new NullPointerException("The SetOfPoints to copy can't be null");
        }
        this.domain = sop.domain;
        this.points = new ArrayList<>(sop.getPoints());
    }

    /**
     * @return domain
     */
    public final Domain getDomain() {
        return domain;
    }

    /**
     * @return ArrayList of points
     */
    public final ArrayList<T> getPoints() {
        return new ArrayList<T>(points);
    }

    /**
     * Add a point to the set
     * @param point point to add
     * @return true if success, false otherwise
     * @throws IllegalArgumentException if the point doesn't belong to the domain
     * @throws NullPointerException if the point is null
     */
    public final boolean add(final T point) throws IllegalArgumentException, NullPointerException {
        if (point == null) {
            throw new NullPointerException("Point to add can't be null");
        }
        if (domain.contains(point))
            return points.add(point);
        else
            throw new IllegalArgumentException("Incompatible point");
    }

    /**
     * Add some point to the set
     * @param points points to add
     * @return true if success, false otherwise
     * @throws IllegalArgumentException if some point doesn't belong to the domain
     * @throws NullPointerException if some point is null
     */
    public final boolean addAll(final Collection<T> points) throws IllegalArgumentException, NullPointerException {
        boolean result = true;
        for (T point : points) {
            result &= this.add(point);
        }
        return result;
    }

    /**
     * Gets the size of the set
     * @return size
     */
    public final int size() {
        return points.size();
    }

    /**
     * Gets the set of points as array
     * @return array of generic points
     */
    public final T[] toArray() {
        @SuppressWarnings("unchecked")
        T[] array = (T[]) Array.newInstance(Point.class, points.size());
        return points.toArray(array);
    }
}
