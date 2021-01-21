import net.jcip.annotations.Immutable;

/**
 * Represents a point in the RGB space. The value of R, G and B can be float
 */
@Immutable
public class RGBPoint extends Point{
    /**
     * Constructor
     * @param r red value
     * @param g green value
     * @param b blue value
     * @throws IllegalArgumentException if one of the parameters is not in [0, 255]
     */
    public RGBPoint(final float r, final float g, final float b) throws IllegalArgumentException {
        super(new float[]{r, g, b});

        if (r < 0 || g < 0 || b < 0 || r > 255 | g > 255 || b > 255) {
            throw new IllegalArgumentException("All colors must be in [0, 255]");
        }
    }

    /**
     * @return red value
     */
    public final float getR(){
        return super.getCoordinate(1);
    }

    /**
     * @return green value
     */
    public final float getG() {
        return super.getCoordinate(2);
    }

    /**
     * @return blue value
     */
    public final float getB() {
        return super.getCoordinate(3);
    }
}
