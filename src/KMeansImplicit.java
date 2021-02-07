import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

import static java.lang.Math.abs;

/**
 * For k-Means clusterization
 * @param <T> subclass of Point
 */
public final class KMeansImplicit<T extends Point> {
    /**
     * Maximum change of a centroid (in every direction) to be considered unchanged
     */
    public static final float TOLERANCE = 0.005F;

    /**
     * Number of threads used by KMeans
     */
    public static final int NUMBER_OF_THREADS = Runtime.getRuntime().availableProcessors();

    /**
     * Performs the k-means clusterization
     * @param k number of clusters
     * @param data points to be clusterized
     * @return clusters
     * @throws IllegalArgumentException if there aren't enough points (<k)
     * @throws NullPointerException if input data is null
     */
    public ArrayList<Cluster<T>> clusterize(int k, final SetOfPoints<T> data) throws IllegalArgumentException, NullPointerException {
        if(data == null) {
            throw new NullPointerException("Input data can't be null");
        }

        int numPoints = data.size();
        if (numPoints < k) {
            throw new IllegalArgumentException("Not enough points for this k (k=" + k + ")");
        }
        if (k == 1) {
            ArrayList<Cluster<T>> clusters = new ArrayList<>(k);
            clusters.add(new Cluster<>(data));
            return clusters;
        }
        ExecutorService ex = Executors.newFixedThreadPool(NUMBER_OF_THREADS);

        final T[] points = data.toArray();

        Point[] centroids = initialCentroids(k, points, ex);
        Integer[] clusterization = new Integer[numPoints];
        boolean stop = false;

        // NewCentroids initialization
        Future<?>[] newCentroidsThreads = new Future[NUMBER_OF_THREADS];
        NewCentroidsTask[] newCentroidsTasks = new NewCentroidsTask[NUMBER_OF_THREADS];
        for (int i = 0; i < NUMBER_OF_THREADS; i++) {
            newCentroidsTasks[i] = new NewCentroidsTask(points, clusterization, k, i);
        }
        int dimension = points[0].getDimension();

        while (!stop) {
            // UpdateClusters
            Point[] actualCentroids = centroids;
            Arrays.parallelSetAll(clusterization, index -> {
                float minDistance = Float.POSITIVE_INFINITY;
                Integer nearestCentroid = null;

                for (int c = 0; c < actualCentroids.length; c++) {
                    float distance = Point.getSquaredEuclideanDistance(actualCentroids[c], points[index]);
                    // cannot throws exception because we ensure that centroids and points wer all not null and of the same dimension
                    if (distance < minDistance) {
                        minDistance = distance;
                        nearestCentroid = c;
                    }
                }
                return nearestCentroid;
            });

            // NewCentroids
            float[][] sum = new float[k][dimension];
            int[] clustersSize = new int[k];
            for (int i = 0; i < NUMBER_OF_THREADS; i++) {
                newCentroidsTasks[i].reset(clustersSize, sum);
                newCentroidsThreads[i] = ex.submit(newCentroidsTasks[i]);
            }
            for (int i = 0; i < NUMBER_OF_THREADS; i++) {
                try {
                    newCentroidsThreads[i].get();
                } catch (InterruptedException | ExecutionException e) {
                    e.printStackTrace();
                }
            }
            Point[] newCentroids = new Point[k];
            for (int w = 0; w < k; w++) {
                float[] coordinate = new float[dimension];
                for (int j = 0; j < dimension; j++) {
                    coordinate[j] = sum[w][j]/ ((float) clustersSize[w]);
                }
                newCentroids[w] = new Point(coordinate);
            }

            // CheckStop
            if(checkStop(centroids, newCentroids)) {
                stop = true;
            } else {
                centroids = newCentroids;
            }
        }

        ex.shutdown();

        ArrayList<Cluster<T>> clusters = new ArrayList<>();
        for (int j = 0; j < k; j++) {
            clusters.add(j, new Cluster<>(data.getDomain()));
        }
        for (int i = 0; i < points.length; i++) {
            clusters.get(clusterization[i]).add(points[i]);
            // cannot throws exception because the domain of all clusters is the same from which the data belongs to
        }

        return clusters;
    }

    /**
     * Checks the stop condition based on the unchange (under a certain tolerance) of centroids position
     * @param oldCentroids old centroids
     * @param newCentroids new centroids
     * @return true if have to stop
     */
    private boolean checkStop(final Point[] oldCentroids, final Point[] newCentroids) {
        for (int k = 0; k < oldCentroids.length; k++) {
            float[] oldCentroid = oldCentroids[k].getCoordinates();
            float[] newCentroid = newCentroids[k].getCoordinates();
            for (int i = 0; i < oldCentroid.length; i++) {
                if (abs(oldCentroid[i] - newCentroid[i]) > KMeansImplicit.TOLERANCE) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Determines the initial centroids by picking them randomly
     * @param k number of centroids
     * @param points points
     * @return centroids
     */
    private Point[] randomInitialCentroids(int k, final ArrayList<T> points) {
        Point[] centroids = new Point[k];
        int numPoints = points.size();
        Random r = new Random();
        for (int i = 0; i < k; i++) {
            int index = r.nextInt(numPoints);
            centroids[i] = new Point(points.get(index));
        }
        return centroids;
    }

    /**
     * Determines the initial centroids by picking the first point in the list and then picking iteratively
     * the point that maximize the minimum distance from previous centroids
     * @param k number of centroids
     * @param points points
     * @return centroids
     */
    private Point[] initialCentroids(int k, final Point[] points, ExecutorService ex) {
        int numPoints = points.length;
        if (numPoints == k) {
            Point[] centroids = new Point[k];
            for (int i = 0; i < k; i++) {
                centroids[i] = new Point(points[i]);
            }

            return centroids;
        }

        Point[] centroids = new Point[k];
        // Random r = new Random();
        // int firstCentroidIndex = r.nextInt(numPoints);
        int firstCentroidIndex = 0;
        centroids[0] = new Point(points[firstCentroidIndex]);

        Future<?>[] threads = new Future[NUMBER_OF_THREADS];
        InitialCentroidsTask[] tasks = new InitialCentroidsTask[NUMBER_OF_THREADS];
        for (int j = 0; j < NUMBER_OF_THREADS; j++) {
            tasks[j] = new InitialCentroidsTask(centroids, points, j);
        }

        AtomicInteger maxMinDistance = new AtomicInteger();
        AtomicInteger newCentroidIndex = new AtomicInteger();

        for (int i = 1; i < k; i++) {
            maxMinDistance.set(0);
            newCentroidIndex.set(0);

            for (int j = 0; j < NUMBER_OF_THREADS; j++) {
                tasks[j].nextIteration(maxMinDistance, newCentroidIndex);
                threads[j] = ex.submit(tasks[j]);
            }
            for (int j = 0; j < NUMBER_OF_THREADS; j++) {
                try {
                    threads[j].get();
                } catch (InterruptedException | ExecutionException e) {
                    e.printStackTrace();
                }
            }

            centroids[i] = new Point(points[newCentroidIndex.get()]);
        }

        return centroids;
    }
}
