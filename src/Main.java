import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class Main {
    private static final int K_MIN = 25;
    private static final int K_MAX = 25;
    private static final int K_STEP = 2;
    private static final int REPETITIONS = 2;

    public static void main(String[] args) {
        try {
            int[] keys = new int[K_MAX - K_MIN + 1];
            double[] times = new double[K_MAX - K_MIN + 1];
            String filename = "3K";

            BufferedImage img = Image.load("src/image/" + filename + ".jpg");
            SetOfPoints<RGBPixel> data = Image.pixelize(img);

            KMeans<RGBPixel> kmeans = new KMeans<>();
            //KMeansImplicit<RGBPixel> kmeans = new KMeansImplicit<>();

            for (int k = K_MIN; k <= K_MAX; k += K_STEP) {
                ArrayList<Cluster<RGBPixel>> clusters = null;
                double diffTimeMillis = 0;
                for (int i = 0; i < REPETITIONS; i++) {
                    clusters = null;        // to garbage collect previous result and avoid Heap Space Error

                    long startTimeMillis = System.currentTimeMillis();
                    clusters = kmeans.clusterize(k, data);
                    long endTimeMillis = System.currentTimeMillis();

                    diffTimeMillis += endTimeMillis - startTimeMillis;
                    System.out.println("For k = " + Integer.toString(k) + ", computation time is " + Double.toString((endTimeMillis-startTimeMillis)/1000d) + " seconds");
                }

                Image.export(clusters, "src/image/" + filename + "-reduced" + k + ".png", img.getWidth(), img.getHeight());

                double meanTimeSeconds = diffTimeMillis / (1000d * REPETITIONS);
                keys[k - K_MIN] = k;
                times[k - K_MIN] = meanTimeSeconds;
                System.out.println("For k = " + Integer.toString(k) + ", MEAN computation time is " + Double.toString(meanTimeSeconds) + " seconds");
            }

            saveTextFile(keys, times, "timesOf" + filename);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void saveTextFile(final int[] kList, final double[] timeList, String fileName) {
        String path = "src/image/" + fileName + ".txt";
        try {
            File file = new File(path);
            BufferedWriter output = new BufferedWriter(new FileWriter(file));
            for (int k = 0; k < kList.length; k += K_STEP) {
                output.write(Integer.toString(kList[k]));
                output.write(" ");
                output.write(Double.toString(timeList[k]));
                output.write("\n");
            }
            output.close();
        } catch ( IOException e ) {
            e.printStackTrace();
        }
    }

}
