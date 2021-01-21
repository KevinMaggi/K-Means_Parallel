import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class Main {
    private static final int K_MIN = 1;
    private static final int K_MAX = 3;

    public static void main(String[] args) {
        try {
            int[] keys = new int[K_MAX - K_MIN + 1];
            double[] times = new double[K_MAX - K_MIN + 1];
            String filename = "6K";

            BufferedImage img = Image.load("src/image/" + filename + ".jpg");
            SetOfPoints<RGBPixel> data = Image.pixelize(img);

            KMeans<RGBPixel> kmeans = new KMeans<>();

            for (int k = K_MIN; k <= K_MAX; k++) {
                long startTimeMillis = System.currentTimeMillis();
                ArrayList<Cluster<RGBPixel>> clusters = kmeans.clusterize(k, data);
                long endTimeMillis = System.currentTimeMillis();

                Image.export(clusters, "src/image/" + filename + "-reduced" + k + ".png", img.getWidth(), img.getHeight());

                double diffTimeSeconds = (double) (endTimeMillis - startTimeMillis) / 1000;
                keys[k - K_MIN] = k;
                times[k - K_MIN] = diffTimeSeconds;
                System.out.println("For k = " + Integer.toString(k) + ", computation time is " + Double.toString(diffTimeSeconds) + " seconds");
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
            for (int k = 0; k < kList.length; k++) {
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
