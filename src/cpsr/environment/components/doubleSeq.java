package cpsr.environment.components;

import java.io.Serializable;
import java.util.Arrays;

public class doubleSeq implements Serializable {
	private static final long serialVersionUID = 5547389585191736210L;
	int FirstHash = 0;
	int SecondHash;
	public doubleSeq(double[] val) {
		int[] val1 = new int[val.length];
		for (int i = 0; i < val.length; i++)
		{
			val1[i] = Long.valueOf(Double.doubleToLongBits(val[i])).hashCode();
		}
		FirstHash = FirstHashCode(val1);
		SecondHash = Arrays.toString(val).hashCode();
	}
	public int getSecondHash()
	{
		return SecondHash;
	}
	public int getFirstHash()
	{
		return FirstHash;
	}
	public int FirstHashCode(int[] val) {
		int h = 7;
        int length = val.length >> 1;
        for (int i = 0; i < length; i++) {
            h = 31 * h + val[ i];
        }
        return h;
	}

	public int hashCode() {
		return SecondHash + FirstHash;
	}
	public boolean equals(Object obj) {
		if (this == obj) {
			return true;
		}
		if  (!(obj instanceof doubleSeq)) {
			return false;
		}
		doubleSeq other = (doubleSeq)obj;
		if (other.getSecondHash()!=this.getSecondHash()||other.getFirstHash()!=this.getFirstHash())
		{
			return false;
		}
		return true;
	}
	public String toString() {
		return Integer.toString(FirstHash) + "," + Integer.toString(SecondHash);
	}
}
