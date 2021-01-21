public class KMeansIterationBlock implements Runnable {
    private final Point[] centroids;    // only read
    private final Point[] points;       // only read
    private final Integer[] clusterization; // shared (but every thread accesses ONLY a LARGE number of CONSECUTIVE elements)
    private final int from;
    private final int to;

    public KMeansIterationBlock(final Point[] centroids, final Point[] points, Integer[] clusterization, int from, int to) {
        this.centroids = centroids;
        this.points = points;
        this.clusterization = clusterization;
        this.from = from;
        this.to = to;
    }

    @Override
    public void run() {
        for (int p = from; p <= to; p++) {
            float minDistance = Float.POSITIVE_INFINITY;
            Integer nearestCentroid = null;

            for (int c = 0; c < centroids.length; c++) {
                float distance = Point.getEuclideanDistance(centroids[c], points[p]);
                    // cannot throws exception because we ensure that centroids and points wer all not null and of the same dimension
                if (distance < minDistance) {
                    minDistance = distance;
                    nearestCentroid = c;
                }
            }
            clusterization[p] = nearestCentroid;
        }
    }
}