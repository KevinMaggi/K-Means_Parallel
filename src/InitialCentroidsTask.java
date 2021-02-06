import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

public class InitialCentroidsTask implements Runnable {
    /**
     * Actual centroids.
     * This array is SHARED, but it is accessed ONLY for READS
     */
    private final Point[] centroids;

    /**
     * Dataset points.
     * This array is SHARED, but it is accessed ONLY for READS
     */
    private final Point[] points;

    /**
     * Actual iteration over the number of centroids
     */
    private int iteration = 0;

    /**
     * Index of the new centroid
     */
    private AtomicInteger newCentroidIndex = null;

    /**
     * Maximum minimum distance
     */
    private AtomicInteger maxMinDistance = null;

    /**
     * Lock for critical section (the final comparison between the private candidate for new centroid)
     */
    private static ReentrantLock lock = new ReentrantLock(true);

    /**
     * First index of the block
     */
    private final int from;
    /**
     * Last index of the block
     */
    private final int to;

    public InitialCentroidsTask(final Point[] centroids, final Point[] points, int index) {
        this.centroids = centroids;
        this.points = points;

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
     * Reset the parameter (shared between tasks) for a new iteration.
     * @param maxMinDistance maximum minimum distance
     * @param newCentroidIndex index of the new centroid
     */
    public void nextIteration(AtomicInteger maxMinDistance, AtomicInteger newCentroidIndex) {
        this.iteration++;
        this.maxMinDistance = maxMinDistance;
        this.newCentroidIndex = newCentroidIndex;
    }

    @Override
    public void run() {
        int candidateCentroid = 0;
        float candidateMaxMinDistance = 0;
        for (int p = from; p <= to; p++) {
            float minDistance = Float.POSITIVE_INFINITY;
            for (int c = 0; c < iteration; c++) {
                float distance = Point.getSquaredEuclideanDistance(points[p], centroids[c]);
                // cannot throws exception because SetOfPoint ensure that all the points are not null and of the same dimension
                if (distance < minDistance) {
                    minDistance = distance;
                }
            }
            if (minDistance > candidateMaxMinDistance) {
                candidateMaxMinDistance = minDistance;
                candidateCentroid = p;
            }
        }

        // Compare with other candidates
        lock.lock();
            if (candidateMaxMinDistance > Float.intBitsToFloat(maxMinDistance.get())) {
                maxMinDistance.set(Float.floatToIntBits(candidateMaxMinDistance));
                newCentroidIndex.set(candidateCentroid);
            }
        lock.unlock();
    }
}