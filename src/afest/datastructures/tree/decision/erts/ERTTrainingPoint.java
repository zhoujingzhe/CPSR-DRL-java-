/*
 *   Copyright 2011 Guillaume Saulnier-Comte
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */

package afest.datastructures.tree.decision.erts;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import org.jblas.DoubleMatrix;

import Parameter.Param;
import afest.datastructures.tree.decision.interfaces.ITrainingPoint;
import cpsr.environment.components.doubleSeq;
import cpsr.model.MemEffCPSR_NotCompressingHistory.MaxSizeHashMap;
import cpsr.model.components.PredictionVector;


/**
 * Class representing an ERTPoint using strings as feature names.
 * @param <O> Type of data contained in the ERTPoint.
 */
public class ERTTrainingPoint<O extends Serializable> extends ERTPoint implements ITrainingPoint<String, O>
{

	private static final long serialVersionUID = 1L;
	
	private O fContent;

	static Map<doubleSeq, Object> ERTLists = Collections.synchronizedMap(new HashMap<doubleSeq, Object>());

	public static int getERTListsSize()
	{
		return ERTLists.size();
	}
	
	public static void removeContentERTList()
	{
		ERTLists.clear();
		System.gc();
//		System.out.println("After clear ERTList:" + ERTLists.size());
	}
	@Override
	public boolean equals(Object obj) {
		// TODO Auto-generated method stub
		ERTTrainingPoint<O> fobj = (ERTTrainingPoint<O>) obj;
		double var1 = (double) this.fContent;
		double var2 = (double) fobj.getContent();
		var1 = ((double)(Math.round(var1 * 1000000.0))) / 1000000.0;
		var2 = ((double)(Math.round(var2 * 1000000.0))) / 1000000.0;
		if (var1 == var2)
		{
			HashMap<String, Double> fValue1 = this.getfValues();
			HashMap<String, Double> fValue2 = fobj.getfValues();
			for (String key:fValue1.keySet())
			{
				double var3 = fValue1.get(key);
				double var4 = fValue2.get(key);
				var3 = ((double)(Math.round(var3 * 1000000.0))) / 1000000.0;
				var4 = ((double)(Math.round(var4 * 1000000.0))) / 1000000.0;
				if (var3 != var4)
				{
					return false;
				}
			}
			return true;
		}
		return false;
	}
	/*
	 * generate nodes used for constructing trees, instead of using randvec to generate index, directly using String.hashCode
	 * There are some hashCrash. To overcome it, I add a list to contain the crashed hashblocks.
	 * When return a block, I compare it with other elements.
	 */
	private static <O extends Serializable> ERTTrainingPoint<O> BuildingERTTrainingPoint(O content, DoubleMatrix pv) throws Exception
	{
		double[] pval = pv.toArray();
		double[] dval = new double[pval.length + 1];
		for (int i = 0; i < pval.length; i++)
		{
			dval[i] = pval[i];
		}
		dval[pval.length] = (double) content;
		doubleSeq index = new doubleSeq(dval);
		HashMap<String, Double> values = InitializeFeatureListForPV(pv.toArray());
		ERTTrainingPoint<O> point= new ERTTrainingPoint<O>(values, content);
		Object obj = null;
		do {
			if (ERTLists.containsKey(index))
			{
				obj = ERTLists.get(index);
				if (point.equals(obj))
				{
					return (ERTTrainingPoint<O>) obj;
				}
				ERTTrainingPoint<O> point1 = (ERTTrainingPoint<O>) obj;
				double[] pv2 = new double[point1.getfValues().size() + dval.length + 1];
				int i = 0;
				for (int j = 0; j < dval.length; j++)
				{
					pv2[i++] = dval[j];
				}
				for (String idx : point1.getfValues().keySet())
				{
					pv2[i++] = point1.getfValues().get(idx);
				}
				pv2[i] = (double) point1.getContent();
				index = new doubleSeq(pv2);
				dval = pv2;
			}
			else
			{
				synchronized (ERTLists) {
					if (!ERTLists.containsKey(index))
					{
						ERTLists.put(index, point);
					}
				}
			}
		}while(true);
	}
	
	/**
	 * Create a ERTPoint with the corresponding values and feature names.
	 * @param values values of the features.
	 * @param content content of the data point.
	 */
	private ERTTrainingPoint(HashMap<String, Double> values, O content)
	{	
		
		super(values);
		fContent = content;
	}
	
	@Override
	public O getContent()
	{
		return fContent;
	}

	/**
	 * Return an ArrayList of ERTPoints created using the given HashMap of features and labels (point are in the same order as the MultiSignal).
	 * @param <O> Type of contents to be placed in the ERTPoints.
	 * @param features HashMap containing the features to create the ERTPoints with.
	 * @param contents contents used for the ERTPoints (must be in the same order as the feature values).
	 * @return an ArrayList of ERTPoints created using the given MultiSignal and labels.
	 */
	public static <O extends Serializable> ArrayList<ERTTrainingPoint<O>> getERTPoints(ArrayList<PredictionVector>features, 
																					   ArrayList<ArrayList<O>> contents)
	{
		ArrayList<ERTTrainingPoint<O>> points = new ArrayList<ERTTrainingPoint<O>>();
		int secondIndex = 0;
		int firstIndex = 0;
		for (int i = 0; i < features.size(); i++)
		{
			ERTTrainingPoint<O> point = null;
			try {
				point = BuildingERTTrainingPoint(contents.get(firstIndex).get(secondIndex), features.get(i).getVector());
			}catch(Exception e)
			{
				System.err.println("Error on getERTPoints");
			}
			secondIndex = (secondIndex+1)%contents.get(firstIndex).size();
			if (secondIndex == 0)
			{
				firstIndex++;
			}
			points.add(point);
		}
		if (firstIndex != contents.size())
		{
			System.out.println("Unsuccessful on getERTPoints");
		}
		return points;
	}
}
