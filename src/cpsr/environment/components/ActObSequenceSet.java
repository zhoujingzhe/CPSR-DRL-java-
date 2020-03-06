package cpsr.environment.components;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.jblas.DoubleMatrix;

import Parameter.Param;

public class ActObSequenceSet implements Serializable
{
	@Override
	public Object clone() throws CloneNotSupportedException {
		ActObSequenceSet cloneset = new ActObSequenceSet();
		return cloneset;
	}
	/**
	 * 
	 */
	private static final long serialVersionUID = -7022506482538122868L;
	private Map<IntSeq, Integer> indexMap = Collections.synchronizedMap(new HashMap<IntSeq, Integer>());
	private Map<IntSeq, List<ActionObservation>> tests = Collections.synchronizedMap(new HashMap<IntSeq, List<ActionObservation>>());
	public Map<Integer, List<ActionObservation>> InverseTests = Collections.synchronizedMap(new HashMap<Integer, List<ActionObservation>>());
	public List<ActionObservation> getTestByCounter(int counter)
	{
		return InverseTests.get(counter);
	}
	private int counter = 0;
	public ActObSequenceSet()
	{
	}

	public void addActObSequence(List<ActionObservation> actobs)
	{
		List<ActionObservation> tmp = new ArrayList<ActionObservation>(actobs);
		while(true)
		{
			IntSeq id = computeID(actobs);
			if(!indexMap.containsKey(id))
			{
				synchronized (indexMap) {
					if (!indexMap.containsKey(id))
					{
						indexMap.put(id, counter);
						tests.put(id, new ArrayList<ActionObservation> (tmp));
						InverseTests.put(counter, new ArrayList<ActionObservation> (tmp));
						counter++;
					}
				}
			}
			else
			{
				List<ActionObservation> t = tests.get(id);
				if (t.equals(actobs))
				{
					return ;
				}
				actobs.addAll(tests.get(id));
			}
			if (indexMap.size() != counter)
			{
				System.err.println("Error On ActObSequenceSet!");
			}
		}
	}
	
	public int indexOf(List<ActionObservation> actobs)
	{
		List<ActionObservation> tmp = new ArrayList<ActionObservation>(actobs);
		while(true)
		{
			IntSeq id = computeID(actobs);
			Integer index =  indexMap.get(id);
			List<ActionObservation> test = tests.get(id);
			if(index == null)
			{
				return -1;
			}
			if (tmp.equals(test))
			{
				return index;
			}
			actobs.addAll(test);
		}
	}
	
	public int size()
	{
		return indexMap.size();
	}
	public Map<IntSeq, Integer> GetindexMap()
	{
		return indexMap;
	}
	
//	public void MergeIndexMap(Map<IntSeq, Integer> IdMap)
//	{
//		for (IntSeq idx : IdMap.keySet())
//		{
//			if(!indexMap.containsKey(idx))
//			{
//				indexMap.put(idx, counter);
//				counter++;
//			}
//			else
//			{
//				if (!indexMap.get(idx).equals(IdMap.get(idx)))
//				{
//					System.err.println("The two different Objects have a same id!");
//				}
//			}
//		}
//		if (indexMap.size() != counter)
//		{
//			System.err.println("Error On ActObSequenceSet!");
//		}
//	}
	
	public static IntSeq computeID(List<ActionObservation> actobs)
	{
		int[] tmp1 = new int[actobs.size()];
		for (int rowidx = 0; rowidx < actobs.size(); rowidx++)
		{
			tmp1 [rowidx] = actobs.get(rowidx).getID();
		}
		return new IntSeq(tmp1);
	}
	
	/*
	private int computeID(List<ActionObservation> actobs)
	{
//		int iteration = 0;
		int id = 0;
		DoubleMatrix tmp = DoubleMatrix.zeros(actobs.size(), 1);
		for(int rowidx = 0; rowidx < actobs.size(); rowidx++)
		{
			tmp.put(rowidx, 0, (int)actobs.get(rowidx).getID());
//			tmp[rowidx] =  actobs.get(rowidx).getID();
//			iteration++;
		}
		return tmp;
	}
	*/
}
