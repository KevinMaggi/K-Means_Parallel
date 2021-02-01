public class KMeansIterationBlock implements Runnable {
    private final Point[] centroids;    // only read
    private final Point[] points;       // only read
    private final Integer[] clusterization; // shared (but every thread accesses ONLY a LARGE number of CONSECUTIVE elements)
    private final int index;

    public KMeansIterationBlock(final Point[] centroids, final Point[] points, Integer[] clusterization, int index) {
        this.centroids = centroids;
        this.points = points;
        this.clusterization = clusterization;
        this.index = index;
    }

    @Override
    public void run() {
        int numPoints = points.length;
        int blockDim = numPoints/KMeans.NUMBER_OF_THREADS + ((numPoints % KMeans.NUMBER_OF_THREADS == 0) ? 0 : 1); // ceil
        int from = blockDim * index;
        int to;
        if (index != KMeans.NUMBER_OF_THREADS - 1) {
            to = blockDim * (index + 1) - 1;
        } else {
            to = numPoints - 1;
        }

        for (int p = from; p <= to; p++) {
            float minDistance = Float.POSITIVE_INFINITY;
            Integer nearestCentroid = null;

            for (int c = 0; c < centroids.length; c++) {
                float distance = Point.getSquaredEuclideanDistance(centroids[c], points[p]);
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