import java.util.Collection;

/**
 * Represents a cluster of points
 */
public class Cluster<T extends Point> extends SetOfPoints<T> {
    /**
     * Constructor
     * @param domain domain to which the points belong
     * @throws NullPointerException if the domain is null
     */
    public Cluster(Domain domain) throws NullPointerException {
        super(domain);
    }

    /**
     * Constructor
     * @param domain domain to which the points belong
     * @param points points to collect
     * @throws NullPointerException if the domain is null
     */
    public Cluster(Domain domain, Collection<T> points) throws NullPointerException {
        super(domain, points);
    }

    /**
     * Constructor. As a Cluster IS a SetOfPoints and a Cluster doesn't have additional attributes, it works also as copy constructor
     * @param sop Set Of Points
     * @throws NullPointerException if the set of point is null
     */
    public Cluster(SetOfPoints<T> sop) throws NullPointerException {
        super(sop);
    }

    /**
     * Get the centroid of the cluster
     * @return centroid
     */
    public Point getCenter() {
        int dimension = getDomain().getDimension();
        float[] coordinates = new float[dimension];

        for (int i = 0; i < dimension; i++) {
            for (T p : this.points) {
                coordinates[i] += p.getCoordinate(i + 1);
                    // cannot throws exception because we know the dimension and we respect it
            }
            coordinates[i] /= this.points.size();
        }

        return new Point(coordinates);
    }

    /**
     * Get the diameter of the cluster
     * @return diameter
     */
    public float getDiameter() {
        float max = 0;
        int n = this.points.size();
        for (int i = 0; i < n; i++) {
            Point p1 = this.points.get(i);
            for (int j = i+1; j < n; j++) {
                Point p2 = this.points.get(j);
                float distance = Point.getEuclideanDistance(p1, p2);
                    // cannot throws exception because points from a SetOfPoint is secure that have same dimension
                if (distance > max) {
                    max = distance;
                }
            }
        }
        return max;
    }
}
