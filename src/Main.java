import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class Main {
    /**
     * Min value of K to test
     */
    private static final int K_MIN = 2;
    /**
     * Max value of K to test
     */
    private static final int K_MAX = 20;
    /**
     * Step on values of K
     */
    private static final int K_STEP = 6;

    /**
     * Image dimension to test: 4K, 5K or 6K
     */
    private static final String IMAGE_DIMENSION = "6K";
    /**
     * Number of image of each dimension to test (max 3)
     */
    private static final int IMAGE_QUANTITY = 3;

    /**
     * Number of times to test each image
     */
    private static final int REPETITIONS = 2;

    public static void main(String[] args) {
        try {
            int[] keys = new int[K_MAX - K_MIN + 1];
            float[] times = new float[K_MAX - K_MIN + 1];

            // Invert comment on the next two lines to change KMeans implementation
            //KMeans<RGBPixel> kmeans = new KMeans<>();
            KMeansImplicit<RGBPixel> kmeans = new KMeansImplicit<>();

            for (int k = K_MIN; k <= K_MAX; k += K_STEP) {
                long cumulativeTimesMillis = 0;
                for (int imageIndex = 1; imageIndex <= IMAGE_QUANTITY; imageIndex++) {
                    BufferedImage img = Image.load("src/image/" + IMAGE_DIMENSION + "-" + imageIndex + ".jpg");
                    SetOfPoints<RGBPixel> data = Image.pixelize(img);
                    int width = img.getWidth();
                    int height = img.getHeight();
                    img = null;     // to garbage collect previous result and avoid Heap Space Error

                    ArrayList<Cluster<RGBPixel>> clusters = null;
                    for (int i = 0; i < REPETITIONS; i++) {
                        clusters = null;        // to garbage collect previous result and avoid Heap Space Error

                        long startTimeMillis = System.currentTimeMillis();
                        clusters = kmeans.clusterize(k, data);
                        long endTimeMillis = System.currentTimeMillis();

                        cumulativeTimesMillis += endTimeMillis - startTimeMillis;
                        System.out.println("For k = " + k + " on image" + imageIndex + ", computation time is " + (endTimeMillis-startTimeMillis)/1000f + " seconds");
                    }

                    Image.export(clusters, "out/results/" + IMAGE_DIMENSION + "-" + imageIndex + "-quantized" + k + ".png", width, height);
                }
                float meanTimeSeconds = cumulativeTimesMillis / (1000f * REPETITIONS * IMAGE_QUANTITY);
                keys[k - K_MIN] = k;
                times[k - K_MIN] = meanTimeSeconds;
                System.out.println("For k = " + k + ", MEAN computation time is " + meanTimeSeconds + " seconds");
            }

            saveTextFile(keys, times, "timesOf" + IMAGE_DIMENSION);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void saveTextFile(final int[] kList, final float[] timeList, String fileName) {
        String path = "out/results/" + fileName + ".txt";
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
