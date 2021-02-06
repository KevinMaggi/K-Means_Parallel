public class UpdateClustersTask implements Runnable {
    /**
     * Actual centroids.
     * This array is SHARED, but it is accessed ONLY for READS
     */
    private Point[] centroids;

    /**
     * Dataset points.
     * This array is SHARED, but it is accessed ONLY for READS
     */
    private final Point[] points;

    /**
     * Actual clusterization.
     * This array is SHARED, but every thread accesses ONLY a LARGE number of CONSECUTIVE elements of it
     */
    private final Integer[] clusterization;

    /**
     * First index of the block
     */
    private final int from;
    /**
     * Last index of the block
     */
    private final int to;

    public UpdateClustersTask(final Point[] centroids, final Point[] points, Integer[] clusterization, int index) {
        this.centroids = centroids;
        this.points = points;
        this.clusterization = clusterization;

        int numPoints = points.length;
        int blockDim = (int) Math.ceil((double) numPoints/KMeans.NUMBER_OF_THREADS);
        from = blockDim * index;
        if (index != KMeans.NUMBER_OF_THREADS - 1) {
            to = blockDim * (index + 1) - 1;
        } else {
            to = numPoints - 1;
        }
    }

    public void updateCentroids(final Point[] centroids) {
        this.centroids = centroids;
    }

    @Override
    public void run() {
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