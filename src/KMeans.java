import java.util.*;
import java.util.concurrent.*;

import static java.lang.Math.abs;

/**
 * For k-Means clusterization
 * @param <T> subclass of Point
 */
public final class KMeans<T extends Point> {
    /**
     * Maximum change of a centroid (in every direction) to be considered unchanged
     */
    public static final float tolerance = 0.005F;

    private final int NUMBER_OF_THREADS = Runtime.getRuntime().availableProcessors() + 1;

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

        T[] points = data.toArray();
        int blockDim = numPoints/NUMBER_OF_THREADS + ((numPoints % NUMBER_OF_THREADS == 0) ? 0 : 1); // ceil

        Point[] centroids = initialCentroids(k, data.getPoints());
        Integer[] clusterization = new Integer[numPoints];
        boolean stop = false;

        while (!stop) {
            Future<?>[] tasks = new Future[NUMBER_OF_THREADS];

            for (int i = 0; i < NUMBER_OF_THREADS; i++) {
                int from = blockDim*i;
                int to;
                if (i != NUMBER_OF_THREADS - 1) {
                    to = blockDim*(i+1) - 1;
                } else {
                    to = numPoints - 1;
                }

                tasks[i] = ex.submit(new KMeansIterationBlock(centroids, points, clusterization, from, to));
            }
            for (int i = 0; i < NUMBER_OF_THREADS; i++) {
                try {
                    tasks[i].get();
                } catch (InterruptedException | ExecutionException e) {
                    e.printStackTrace();
                }
            }

            Point[] newCentroids = newCentroids(points, clusterization, k);

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
                if (abs(oldCentroid[i] - newCentroid[i]) > KMeans.tolerance) {
                    return false;
                }
            }
        }
        return true;
    }



    /**
     * Calculates the new centroids based on the updated clusterization
     * @param points points
     * @param clusterization clusterization
     * @param k number of clusters
     * @return new centroids
     */
    private Point[] newCentroids(final Point[] points, final Integer[] clusterization, int k) {
        int dimension = points[0].getDimension();
        float[][] sum = new float[k][dimension];
        int[] clustersSize = new int[k];
        for (int i = 0; i < points.length; i++) {
            for (int j = 0; j < dimension; j++) {
                sum[clusterization[i]][j] += points[i].getCoordinate(j+1);
                    // cannot throws exception because all point has the same dimension and we respect it
            }
            clustersSize[clusterization[i]]++;
        }

        Point[] centroids = new Point[k];
        for (int w = 0; w < k; w++) {
            float[] coordinate = new float[dimension];
            for (int j = 0; j < dimension; j++) {
                coordinate[j] = sum[w][j]/clustersSize[w];
            }
            centroids[w] = new Point(coordinate);
        }

        return centroids;
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
    private Point[] initialCentroids(int k, final ArrayList<T> points) {
        int numPoints = points.size();
        if (numPoints == k) {
            Point[] centroids = new Point[k];
            for (int i = 0; i < k; i++) {
                centroids[i] = new Point(points.get(i));
            }

            return centroids;
        }

        Point[] centroids = new Point[k];
        int[] pointIndexes = new int[k];
        // Random r = new Random();
        // int firstIndex = r.nextInt(numPoints);
        int firstIndex = 0;
        centroids[0] = points.get(firstIndex);
        pointIndexes[0] = firstIndex;

        for (int i = 1; i < k; i++) {
            float maxMinDistance = 0;
            int newCentroidIndex = 0;

            for (int p = 0; p < numPoints; p++) {
                float minDistance = Float.POSITIVE_INFINITY;
                for (int indexCentroid : pointIndexes) {
                    float distance = Point.getEuclideanDistance(points.get(p), points.get(indexCentroid));
                        // cannot throws exception because SetOfPoint ensure that all the points are not null and of the same dimension
                    if (distance < minDistance) {
                        minDistance = distance;
                    }
                }
                if (minDistance > maxMinDistance) {
                    maxMinDistance = minDistance;
                    newCentroidIndex = p;
                }
            }
            centroids[i] = new Point(points.get(newCentroidIndex));
            pointIndexes[i] = newCentroidIndex;
        }

        return centroids;
    }
}
