import net.jcip.annotations.Immutable;

/**
 * Represents a Pixel, so a point in the RGB space with integer value and the coordinates x and y in an image
 */
@Immutable
public class RGBPixel extends RGBPoint{
    /**
     * Position coordinates of the pixel in the image
     */
    private final int x,y;

    /**
     * Constructor
     * @param x x coordinate in the image
     * @param y y coordinate in the image
     * @param r red value
     * @param g green value
     * @param b blue value
     * @throws IllegalArgumentException if one of r, g, b is not in [0, 255] or if one of x, y are negative
     */
    public RGBPixel(int x, int y, int r, int g, int b) throws IllegalArgumentException{
        super(r,g,b);

        if (x < 0 || y < 0) {
            throw new IllegalArgumentException("x and y coordinate can't be negative");
        }
        this.x = x;
        this.y = y;
    }

    /**
     * @return x coordinate
     */
    public final int getX() {
        return x;
    }

    /**
     * @return y coordinate
     */
    public final int getY() {
        return y;
    }
}
