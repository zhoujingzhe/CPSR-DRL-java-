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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import afest.datastructures.tree.decision.erts.exceptions.ERTException;
import afest.datastructures.tree.decision.interfaces.IPoint;
import cpsr.environment.components.doubleSeq;



/**
 * Class representing an ERTPoint using strings as feature names.
 */
public class ERTPoint implements IPoint<String>
{

	private static final long serialVersionUID = 1L;
	
	protected HashMap<String, Double> fValues;
	
	public HashMap<String, Double> getfValues()
	{
		return fValues;
	}
	
	/**
	 * Create a ERTPoint with the corresponding values and feature names.
	 * @param values values of the features.
	 */
	public ERTPoint(HashMap<String, Double> values)
	{	
		fValues = values;
	}
	
	@Override
	public String[] getAttributes()
	{
		String[] names = new String[fValues.size()];
		names = fValues.keySet().toArray(names);
		return names;
	}

	@Override
	public double getValue(String attribute)
	{
		Double value = fValues.get(attribute);
		if (value == null)
		{
			throw new ERTException("Attribute name not present in point! name: "+attribute);
		}
		return value;
	}

	@Override
	public int size()
	{
		return fValues.size();
	}

	/**
	 * Return an ArrayList of ERTPoints created using the given HashMap of features.
	 * @param features HashMap containing the features to create the ERTPoints with.
	 * @return an ArrayList of ERTPoints created using the given features. 
	 */
	public static ArrayList<ERTPoint> getERTPoints(ArrayList<double[]> features)
	{
		ArrayList<ERTPoint> points = new ArrayList<ERTPoint>();
		
		// ugly !!!
		int size = features.size();
		
		for (int i = 0; i < size; i++)
		{
			HashMap<String, Double> values = null;
			try {
				values = InitializeFeatureListForPV(features.get(i));
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			ERTPoint point = new ERTPoint(values);
			points.add(point);
		}
		return points;
	}
	
	public static void clearNodeFeatures()
	{
		Nodefeatures.clear();
		System.gc();
	}
	
	private static Map<doubleSeq, HashMap<String, Double>> Nodefeatures = new HashMap<doubleSeq, HashMap<String,Double>>();
//	public static HashMap<String, Double> InitializeFeatureListForPV(double[] pv) throws Exception
//	{
//		for (int i = 0; i < pv.length; i++)
//		{
//			pv[i] = ((double)(Math.round(pv[i] * 1000000.0))) / 1000000.0;
//		}
//		doubleSeq idx = new doubleSeq(pv);
//		double[] pv3 = pv;
//		do 
//		{
//			if (Nodefeatures.containsKey(idx))
//			{
//				HashMap<String, Double> ret = Nodefeatures.get(idx);
//				double[] pv1 = new double[ret.size()];
//				for (int i = 0; i < ret.size(); i++)
//				{
//					pv1[i] = ret.get(Integer.toString(i+1));
//				}
//				if (pv.length != pv1.length)
//				{
//					throw new Exception("lengths of two predictive State are not equal");
//				}
//				boolean eq = true;
//				for (int i = 0; i < pv.length; i++)
//				{
//					if (pv[i] != (double)pv1[i])
//					{
//						eq = false;
//						break;
//					}
//				}
//				if (eq)
//				{
//					return ret;
//				}
//				double[] pv2 = new double[pv3.length + pv1.length];
//				for (int i = 0; i < pv3.length; i++)
//				{
//					pv2[i] = pv3[i];
//				}
//				for (int i = 0; i < pv1.length; i++)
//				{
//					pv2[pv3.length + i] = (double)pv1[i];
//				}
//				idx = new doubleSeq(pv2);
//				pv3 = pv2;
//			}
//			else
//			{
//				synchronized (Nodefeatures)
//				{
//					HashMap<String, Double> s = new HashMap<String, Double>();
//					for (int index = 0; index < pv.length; index++)
//					{
//						s.put(Integer.toString(index + 1), pv[index]);
//					}
//					Nodefeatures.put(idx, s);
//				}
//
//			}
//		}while(true);
//	}
	
	public static HashMap<String, Double> InitializeFeatureListForPV(double[] pv) throws Exception
	{
		HashMap<String, Double> s = new HashMap<String, Double>();
		for (int index = 0; index < pv.length; index++)
		{
			double round_value = ((double)(Math.round(pv[index] * 1000000.0))) / 1000000.0;
			s.put(Integer.toString(index + 1), round_value);
		}
		return s;
	}
}
