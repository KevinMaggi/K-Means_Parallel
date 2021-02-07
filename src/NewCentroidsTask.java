import java.util.concurrent.locks.ReentrantLock;

public class NewCentroidsTask implements Runnable {
    /**
     * Dataset points.
     * This array is SHARED, but it is accessed ONLY for READS
     */
    private final Point[] points;

    /**
     * Actual clusterization.
     * This array is SHARED, but it is accessed ONLY for READS
     */
    private final Integer[] clusterization;

    /**
     * Number of clusters
     */
    private final int k;

    /**
     * Dimension of the points
     */
    private final int dimension;

    /**
     * Size of each cluster.
     * SHARED, but accessed in critical section
     */
    private int[] clustersSize = null;

    /**
     * Sum of coordinates of points of each cluster.
     * SHARED, but accessed in critical section
     */
    private float[][] sum = null;

    /**
     * Lock for critical section (the final sum on clusters' size and sum)
     */
    private static final ReentrantLock lock = new ReentrantLock(true);

    /**
     * First index of the block
     */
    private final int from;
    /**
     * Last index of the block
     */
    private final int to;

    public NewCentroidsTask(final Point[] points, Integer[] clusterization, int k, int index) {
        this.points = points;
        this.clusterization = clusterization;
        this.k = k;

        this.dimension = points[0].getDimension();

        int numPoints = points.length;
        int blockDim = (int) Math.ceil((double) numPoints/KMeans.NUMBER_OF_THREADS);
        from = blockDim * index;
        if (index != KMeans.NUMBER_OF_THREADS - 1) {
            to = blockDim * (index + 1) - 1;
        } else {
            to = numPoints - 1;
        }
    }

    /**
     * Resets parameters for a new iteration
     * @param clustersSize clusters' size
     * @param sum sum of coordinates of points of each cluster
     */
    public void reset(final int[] clustersSize, final float[][] sum) {
        this.clustersSize = clustersSize;
        this.sum = sum;
    }

    @Override
    public void run() {
        float[][] partialSum = new float[k][dimension];
        int[] partialSize = new int[k];
        for (int i = from ; i <= to; i++) {
            for (int j = 0; j < dimension; j++) {
                partialSum[clusterization[i]][j] = partialSum[clusterization[i]][j] + points[i].getCoordinate(j+1);
                // cannot throws exception because all points have the same dimension and we respect it
            }
            partialSize[clusterization[i]]++;
        }

        lock.lock();
        for (int i = 0; i < k; i++) {
            clustersSize[i] += partialSize[i];
            for (int j = 0; j < dimension; j++) {
                sum[i][j] += partialSum[i][j];
            }
        }
        lock.unlock();
    }
}