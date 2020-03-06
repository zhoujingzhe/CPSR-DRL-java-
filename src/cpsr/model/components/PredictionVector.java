package cpsr.model.components;

import java.io.Serializable;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import org.jblas.DoubleMatrix;

import Parameter.Param;
import cpsr.environment.components.doubleSeq;
import cpsr.model.MemEffCPSR_NotCompressingHistory.MaxSizeHashMap;
import cpsr.model.exceptions.PVException;

@SuppressWarnings("serial")
public class PredictionVector implements Serializable
{
	@Override
	public boolean equals(Object obj) {
		PredictionVector pv = (PredictionVector) obj;
		return this.vector.equals(pv.getVector());
	}
	public PredictionVector clone() throws CloneNotSupportedException {
		PredictionVector pv = new PredictionVector(this.getVector());
		return pv;
	}
	private DoubleMatrix vector;
	private static Map<doubleSeq, PredictionVector> pv_list = Collections.synchronizedMap(new HashMap<doubleSeq, PredictionVector>());
	public static int getSizeOfPVList()
	{
		return pv_list.size();
	}
	public static void ClearPV_List()
	{
		pv_list.clear();
		System.gc();
	}
	public static PredictionVector BuildPredctiveVector(DoubleMatrix Mat)
	{
		double[] pv = Mat.toArray();
		doubleSeq idx = new doubleSeq(pv);
		do{
			if (pv_list.containsKey(idx))
			{
				if (pv_list.get(idx).getVector().equals(Mat))
				{
					return pv_list.get(idx);
				}
				double[] newpv = pv_list.get(idx).getVector().toArray();
				double[] pv1 = new double[pv.length + newpv.length];
				int j = 0;
				for (int i = 0; i < pv.length; i++)
				{
					pv1[j++] = pv[i];
				}
				for (int i = 0; i < newpv.length; i++)
				{
					pv1[j++] = newpv[i];
				}
				idx = new doubleSeq(pv1);
				pv = pv1;
			}
			else
			{
				synchronized (pv_list) {
					if (!pv_list.containsKey(idx))
					{
						pv_list.put(idx, new PredictionVector(Mat));
					}
				}
			}
		}while(true);
	}
	
	public static PredictionVector Add(PredictionVector pv1, PredictionVector pv2) throws Exception
	{
		if (pv1.getVector().length != pv2.getVector().length)
		{
			throw new Exception("The two elements should have the same size!");
		}
		DoubleMatrix vec1 = pv1.getVector();
		DoubleMatrix vec2 = pv2.getVector();
		PredictionVector pv = new PredictionVector(vec1.rows);
		for (int index_row = 0; index_row < vec1.rows; index_row++)
		{
			for (int index_column = 0; index_column < vec1.columns; index_column++)
			{
				pv.getVector().put(index_row, index_column, vec1.get(index_row, index_column) + vec2.get(index_row, index_column));
			}
		}
		return pv;
	}
	
	/**
	 * Constructs a prediction vector of specified size with all
	 * values initialized to zero.
	 * 
	 * @param size Size (length) of prediction vector.
	 */
	private PredictionVector(int size)
	{
		vector = new DoubleMatrix(size, 1);
	}
	
	/**
	 * Constructs a prediction vector from a matrix (vector) passed as argument.
	 * This matrix passed must be vector and have second dimension equal to
	 * 1 or exception thrown.
	 * 
	 * @param vector
	 */
	private PredictionVector(DoubleMatrix vector)
	{
		if(vector.getColumns() != 1 )
		{
			throw new PVException("Constructor requires Matrix with column dimension equal to one (i.e. a vector)");
		}
		this.vector = vector.dup();
	}
	
	/**
	 * Returns an entry of prediction vector.
	 * 
	 * @param index Index of entry to be returned
	 * @return The (double) entry specified by index.
	 */
	public double getEntry(int index)
	{
		return vector.get(index, 0);
	}
	
	/**
	 * Returns the size (length) of prediction vector.
	 * 
	 * @return The size (length) of prediction vector.
	 */
	public int getSize()
	{
		return vector.getRows();
	}
	
	/**
	 * Returns Jama matrix representation of prediction vector
	 * 
	 * @return Jama matrix representation of prediction vector
	 */
	public DoubleMatrix getVector()
	{
		return vector;
	}
}
