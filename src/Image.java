import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;

/**
 * Functoide with all operation that can be applied on an image
 */
public class Image {
    /**
     * Loads an image from filesystem as Java Object
     * @param path path of the image in the filesystem
     * @return image
     * @throws IOException if an error occurs reading the file
     */
    public static BufferedImage load(String path) throws IOException {
        return ImageIO.read(new File(path));
    }

    /**
     * Convert the image in a set of pixel (seen as point)
     * @param img image to pixelize
     * @return set of pixel
     * @throws NullPointerException if the image is null
     */
    public static SetOfPoints<RGBPixel> pixelize(BufferedImage img) throws NullPointerException {
        if (img == null) {
            throw new NullPointerException("Image can't be null");
        }

        ArrayList<RGBPixel> data = new ArrayList<>();
        for (int x = 0; x < img.getWidth(); x++) {
            for (int y = 0; y < img.getHeight(); y++) {
                Color color = new Color(img.getRGB(x, y));
                RGBPixel p = new RGBPixel(x, y, color.getRed(), color.getGreen(), color.getBlue());
                data.add(p);    // cannot throws exception because we ensure that "p" is not null and contained in the domain
            }
        }

        return new SetOfPoints<>(Domain.RGB(), data);
    }

    /**
     * Save on the filesystem as PNG a segmentation of an image
     * @param clusters segments
     * @param path path of the file in the filesystem
     * @param width width of the image
     * @param height height of the image
     * @throws IOException if an error occurs writing the file
     * @throws NullPointerException if the collection of clusters is null
     */
    public static void export(Collection<Cluster<RGBPixel>> clusters, String path, int width, int height) throws IOException, NullPointerException {
        if (clusters == null) {
            throw new NullPointerException("Clusters can't be null");
        }

        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
        for (Cluster<RGBPixel> c : clusters) {
            Point center = c.getCenter();
            Color color = new Color(center.getCoordinate(1)/255, center.getCoordinate(2)/255, center.getCoordinate(3)/255);
                // cannot throw exception because a center of a cluster of RGBPixel has certainly 3 dimensions
            for (RGBPixel p : c.getPoints()) {
                img.setRGB(p.getX(), p.getY(), color.getRGB());
            }
        }

        ImageIO.write(img, "png", new File(path));
    }
}
