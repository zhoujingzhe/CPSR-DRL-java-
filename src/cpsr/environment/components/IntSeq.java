package cpsr.environment.components;

import java.io.Serializable;
import java.util.Arrays;

public class IntSeq implements Serializable{
	@Override
	public int hashCode() {
		return getFirstHash() + SecondHash;
	}
	private static final long serialVersionUID = 1471927170947203151L;
	int[] val=null;
	int h = 0;
	int SecondHash;
	public IntSeq(int[] val) {
		this.val = val;
		SecondHash = Arrays.toString(this.val).hashCode();
	}
	public int getSecondHash()
	{
		return SecondHash;
	}
	public int getFirstHash() {
		if (this.h != 0) {
			return this.h;
		}
		int h = 7;
        int length = val.length >> 1;
        for (int i = 0; i < length; i++) {
            h = 31 * h + val[ i];
        }
        this.h = h;
        return h;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj) {
			return true;
		}
		if  (!(obj instanceof IntSeq)) {
			return false;
		}
		IntSeq other = (IntSeq)obj;
		if (other.getSecondHash()!=this.getSecondHash())
		{
			return false;
		}
		return Arrays.equals(this.val, other.val);
	}
	public String toString() {
		return Arrays.toString(val);
	}
}