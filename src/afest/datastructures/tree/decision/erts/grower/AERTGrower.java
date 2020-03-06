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

package afest.datastructures.tree.decision.erts.grower;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;

import org.apache.commons.math.util.FastMath;

import Parameter.Param;
import jparfor.Functor;
import jparfor.MultiThreader;
import jparfor.Range;
import afest.datastructures.tree.decision.DTNode;
import afest.datastructures.tree.decision.DecisionTree;
import afest.datastructures.tree.decision.Forest;
import afest.datastructures.tree.decision.erts.ERTSplit;
import afest.datastructures.tree.decision.erts.exceptions.ERTException;
import afest.datastructures.tree.decision.erts.informationfunctions.interfaces.IERTScore;
import afest.datastructures.tree.decision.interfaces.ITrainingPoint;


/**
 * Class used to grow Extra Trees.
 * @param <R> Type of attributes contained in the Extra Trees.
 * @param <O> Type of object attached to the points.
 * @param <C> Type of content present in the leafs of the Extra Trees.
 */
public abstract class AERTGrower<R extends Serializable, O extends Serializable, C extends Serializable> implements Serializable
{

	private static final long serialVersionUID = 1L;
	private Integer fK;
	private int fNmin;
	private IERTScore<R, O> fScore;
//	private Random fRandom;
	private static Map<Integer, Random> fRandmMap = Collections.synchronizedMap(new HashMap<Integer, Random>());

	
	/**
	 * Creates a ERTGrower that will create ExtraTrees according to the following parameters.
	 * @param k number of splits to create at each inner node.  If k is null, then sqrt(number of attribute) will be used.
	 * @param nMin a leaf is created if |set| <= n_min.
	 * @param score score object used to calculate split scores.
	 */
	public AERTGrower(Integer k, int nMin, IERTScore<R, O> score)
	{
		fK = k;
		fNmin = nMin;
		fScore = score;
//		fRandom = new Random(0);
	}
	
	public Random UsingfRandom(Integer seed)
	{
		if (fRandmMap.containsKey(seed))
		{
			Random fRandom = fRandmMap.get(seed);
			if (fRandom == null)
			{
				System.err.println("fRandom is null");
			}
			return fRandom;
		}
		else
		{
			synchronized (fRandmMap) 
			{
				if (fRandmMap.containsKey(seed))
				{
					Random fRandom = fRandmMap.get(seed);
					if (fRandom == null)
					{
						System.err.println("fRandom is null");
					}
					return fRandom;
				}
				Random fRandom = new Random(Param.getRandomSeed());
				fRandmMap.put(seed, fRandom);
				return fRandom;
			}
		}
	}
	
	/**
	 * Return an extra tree created from the set of points.
	 * @param <T> Type of ITrainingPoints used by the Extra Trees.
	 * @param set set of points to create the tree from.
	 * @return an extra tree created from the set of points.
	 */
	public <T extends ITrainingPoint<R, O>> DecisionTree<R, C> growERT(Collection<T> set, Random fRandom)
	{
		// get the attributes present in the points.
		T aElement = set.iterator().next();
		ArrayList<R> attributeList = new ArrayList<R>();
		for (R attribute : aElement.getAttributes())
		{
			attributeList.add(attribute);
		}
		// set k to sqrt(number of attributes) if unset
		if (fK == null)
		{
			fK = (int) FastMath.ceil(FastMath.sqrt(attributeList.size()));
		}
		
		// Train the tree
		ArrayList<R> constantAttributes = new ArrayList<R>();
		DTNode<R, C> root = buildAnExtraTree(set, constantAttributes, attributeList, fRandom);
		DecisionTree<R, C> tree = new DecisionTree<R, C>(root);
		return tree;
	}
	
	/**
	 * Return a forest of containing numberOfTrees.
	 * @param <T> Type of ITrainingPoints used by the Extra Trees.
	 * @param set set of points to create the forest from.
	 * @param numberOfTrees number of trees to create in the forest.
	 * @return a forest of containing numberOfTrees.
	 */
	public <T extends ITrainingPoint<R, O>> Forest<R, C> growForest(final Collection<T> set, int numberOfTrees)
	{
		Forest<R, C> forest = new Forest<R, C>();
		
		ArrayList<DecisionTree<R, C>> trees = MultiThreader.foreach(new Range(numberOfTrees), new Functor<Integer, DecisionTree<R, C>>()
				{
					@Override
					public DecisionTree<R, C> function(Integer input)
					{
						Random fRandom = UsingfRandom(input);
						DecisionTree<R, C> tree = null;
						try {
							tree = growERT(set, fRandom);
						}catch (Exception e)
						{
							e.printStackTrace();
							throw new RuntimeException(e);
						}
						return tree;
					}
				});
		
		forest.addAll(trees);
		
		return forest;
	}
	
	/**
	 * Return "k" the number of random features selected at each split.
	 * @return "k" the number of random features selected at each split.
	 */
	public Integer getK()
	{
		return fK;
	}
	
	/**
	 * Return "nMin" (a split is performed if more than nMin points are present and not all labels are the same).
	 * @return "nMin"
	 */
	public int getNmin()
	{
		return fNmin;
	}
	
	/**
	 * Return the scoring function used for the training of the extremely randomized trees.
	 * @return the scoring function used for the training of the extremely randomized trees.
	 */
	public IERTScore<R, O> getScoringFunction()
	{
		return fScore;
	}
	
	/**
	 * Return the root of an ExtraTree created using the set of points.
	 * @param <T> Type of ITrainingPoints used by the Extra Trees.
	 * @param set set of points to create the ExtraTree for.
	 * @param constantAttributes constant attributes in the set of points.
	 * @param attributeList list of all attributes present in each point in the set.
	 * @return the root of an ExtraTree created using the set of points.
	 */
	private <T extends ITrainingPoint<R, O>> DTNode<R, C> buildAnExtraTree(Collection<T> set, 
																		   ArrayList<R> constantAttributes, 
																		   ArrayList<R> attributeList, Random fRandom)
	{
		// get the new set of constant attributes
		ArrayList<R> newConstantAttributes = getConstantAttributes(set, constantAttributes, attributeList);
		
		// Check if we build a leaf or an inner node
		if (isLeaf(set, newConstantAttributes, attributeList))
		{
			DTNode<R, C> leaf = createLeaf(set);
			return leaf;
		}
		else
		{
			DTNode<R, C> innerNode = createInnerNode(set, newConstantAttributes, attributeList, fRandom);
			return innerNode;
		}
	}
	
	/**
	 * Return true if the node should be a leaf, false otherwise.
	 * Leaf if:
	 * 	- |set| < n_min, or
	 *  - all attributes are constant, or
	 *  - all labels are constant
	 * @param <T> Type of ITrainingPoints used by the Extra Trees.
	 * @param set set containing the IPoints for which to create a new node.
	 * @param constantAttributes attributes that are constant in the set.
	 * @param attributeList list of all attributes present in each point in the set.
	 * @return true if the node should be a leaf, false otherwise.
	 */
	private <T extends ITrainingPoint<R, O>> boolean isLeaf(Collection<T> set, ArrayList<R> constantAttributes, ArrayList<R> attributeList)
	{
		// Check if the set size is smaller than n_min
		boolean isNmin = false;
		if (set.size() <= fNmin)
		{
			isNmin = true;
		}

		// Check if all attributes are constant
		boolean areAttributesConstant = false;
		int numAttributes = attributeList.size();
		int numConstantAttributes = constantAttributes.size();
		if (numAttributes == numConstantAttributes)
		{
			areAttributesConstant = true;
		}
		
		// Check if all labels are constant
		boolean areLabelsConstant = true;
		T firstElement = set.iterator().next();
		O firstContent = firstElement.getContent();
		for (T aT : set)
		{
			if (!(firstContent.equals(aT.getContent())))
			{
				areLabelsConstant = false;
				break;
			}
		}
		
		return isNmin || areAttributesConstant || areLabelsConstant;
	}
	
	/**
	 * Return the constant attributes in the set, assuming that the attributes contained in constantAttributes are already constant.
	 * @param <T> Type of ITrainingPoints used by the Extra Trees.
	 * @param set set for which to find the constant attributes.
	 * @param constantAttributes set of attributes that are already constant.
	 * @param attributeList list of all attributes present in each point in the set.
	 * @return the constant attributes in the set, assuming that the attributes contained in constantAttributes are already constant.
	 */
	private <T extends ITrainingPoint<R, O>> ArrayList<R> getConstantAttributes(Collection<T> set, 
																				ArrayList<R> constantAttributes, 
																				ArrayList<R> attributeList)
	{
		ArrayList<R> newConstantAttributes = new ArrayList<R>(constantAttributes);

		T firstT = set.iterator().next();
		// For each possible attribute in the points
		for (R anAttribute : attributeList)
		{
			// If it was not already constant
			if (!newConstantAttributes.contains(anAttribute))
			{
				// verify if it is now constant
				boolean isConstant = true;
				for (T aT : set)
				{
					if (firstT.getValue(anAttribute) != aT.getValue(anAttribute))
					{
						isConstant = false;
						break;
					}
				}
				if (isConstant)
				{
					newConstantAttributes.add(anAttribute);
				}
			}
		}
		return newConstantAttributes;
	}
	
	/**
	 * Return a Leaf node create using the set.
	 * @param <T> Type of ITrainingPoints used by the Extra Trees.
	 * @param set set of points to create the leaf for.
	 * @return a Leaf node create using the set.
	 */
	protected abstract <T extends ITrainingPoint<R, O>> DTNode<R, C> createLeaf(Collection<T> set);
	
	/**
	 * Return an inner node created using the set of points.
	 * @param <T> Type of ITrainingPoints used by the Extra Trees.
	 * @param set set of points to create the inner node for.
	 * @param constantAttributes constant attributes present in the set.
	 * @param attributeList list of all attributes present in each point in the set.
	 * @return an inner node created using the set of points.
	 */
	private <T extends ITrainingPoint<R, O>> DTNode<R, C> createInnerNode(Collection<T> set, 
																		  ArrayList<R> constantAttributes, 
																		  ArrayList<R> attributeList, Random fRandom)
	{
		ArrayList<R> kRandomAttributes = getKRandomAttributes(constantAttributes, attributeList, fRandom);
		SplitResult<R, O, T> ret = getBestSplitAndResults(set, kRandomAttributes, fRandom);
		ERTSplit<R> bestSplit = ret.BestSplit;
		DTNode<R, C> innerNode = new DTNode<R, C>();
		innerNode.setSplit(bestSplit);
		
		// Split the data set according to the best split;
		ArrayList<T> leftSet = ret.LeftResult;
		ArrayList<T> rightSet = ret.RightResult;

		if (rightSet.size() == 0 || leftSet.size() == 0)
		{
			throw new ERTException("Split created an empty set! left = "+leftSet.size()+", right = "+rightSet.size()+", "+bestSplit.toString());
		}
		
		// Create children
		DTNode<R, C> leftChild = buildAnExtraTree(leftSet, constantAttributes, attributeList, fRandom);
		DTNode<R, C> rightChild = buildAnExtraTree(rightSet, constantAttributes, attributeList, fRandom);
		
		// assign them to the innerNode
		innerNode.setLeftChild(leftChild);
		innerNode.setRightChild(rightChild);
		
		return innerNode;
	}
	
	/**
	 * Return k random attributes (non-constant) picked without replacement unless less then k attributes are non-constant.
	 * @param constantAttributes attributes that are constant.
	 * @param attributeList list of all attributes present in each point in the set.
	 * @return k random attributes (non-constant) picked without replacement unless less then k attributes are non-constant.
	 */
	private ArrayList<R> getKRandomAttributes(ArrayList<R> constantAttributes, ArrayList<R> attributeList, Random fRandom)
	{
		ArrayList<R> kRandomAttributes = new ArrayList<R>();
		
		HashSet<R> pickedAttributes = new HashSet<R>(constantAttributes);
		for (int k = 0; k < fK; k++)
		{
			// If all non-constant attributes have been picked and k is not reached yet, start resampling the non-constant attributes.
			if (pickedAttributes.size() == attributeList.size())
			{
				pickedAttributes.clear();
				pickedAttributes.addAll(constantAttributes);
			}
			
			// Count the number of attributes that are available for a pick
			int numNotPicked = attributeList.size() - pickedAttributes.size();
			// get a random attribute
			int randomAttribute = fRandom.nextInt(numNotPicked);
			int count = 0;
			for (R aR : attributeList)
			{
				// If the attribute is not picked
				if (!pickedAttributes.contains(aR))
				{
					// verify if it is the one corresponding to the random pick
					if (count == randomAttribute)
					{
						kRandomAttributes.add(aR);
						pickedAttributes.add(aR);
						break;
					}
					else// increase the count
					{
						count++;
					}
				}
			}
		}
		return kRandomAttributes;
	}
	
	/**
	 * Create a split on the given attribute by choosing a uniformly random threshold in [min, max).

	 * @param <T> Type of ITrainingPoints used by the Extra Trees.
	 * @param set set containing the points in which we choose the threshold.
	 * @param attribute attribute to pick the threshold for.
	 * @return a split on the given attribute by choosing a uniformly random threshold in [min, max).
	 */
	
	private <T extends ITrainingPoint<R, O>> ERTSplit<R> createSplit(Collection<T> set, R attribute, Random fRandom)
	{
		double min = Double.POSITIVE_INFINITY;
		double max = Double.NEGATIVE_INFINITY;
		for (T aT : set)
		{
			double value = aT.getValue(attribute);
			if (value > max)
			{
				max = value;
			}
			if (value < min)
			{
				min = value;
			}
		}
		
		if (Double.isInfinite(max))
		{
			max = Double.MAX_VALUE;
		}
		if (Double.isInfinite(min))
		{
			min = -Double.MAX_VALUE;
		}
		
		max = max - Double.MIN_VALUE;
		min = min + Double.MIN_VALUE;
		double rate = fRandom.nextDouble() * (0.95 - 0);
		double threshold =  (rate * (max-min) + min);
		if (threshold >= max)
		{
			System.err.println("The theshold:" + Double.toString(threshold) + " is larger than max:" + Double.toString(max));
			System.err.println("Changing to:" + Double.toString(threshold - Double.MIN_VALUE));
			System.err.println("The rate:"  + Double.toString(rate));
			threshold -= Double.MIN_VALUE;
		}
		else if (threshold <= min)
		{
			System.err.println("The theshold:" + Double.toString(threshold) + " is less than max:" + Double.toString(min));
			System.err.println("Changing to:" + Double.toString(threshold + Double.MIN_VALUE));
			System.err.println("The rate:"  + Double.toString(rate));
			threshold += Double.MIN_VALUE;
		}
		ERTSplit<R> split = new ERTSplit<R>(attribute, threshold);
		return split;
	}
	
//	private <T extends ITrainingPoint<R, O>> ERTSplit<R> createSplit(Collection<T> set, R attribute, Random fRandom)
//	{
//		double quantile = fRandom.nextDouble()*0.99999;
//		int numCount = (int) (set.size() * quantile);
//		double min = Double.POSITIVE_INFINITY;
//		double max = Double.NEGATIVE_INFINITY;
//		double[] valueArray = new double[set.size()];
//		int index = 0;
//		for (T aT : set)
//		{
//			double value = aT.getValue(attribute);
//			valueArray[index++] = value;
//			if (min > value)
//			{
//				min = value;
//			}
//			if (max < value)
//			{
//				max = value;
//			}
//		}
//		Arrays.sort(valueArray);
//		double threshold =  valueArray[numCount];
//		if (threshold >= max)
//		{
//			threshold = threshold - 0.00000001;
//		}
//		else if (threshold <= min)
//		{
//			threshold = threshold + 0.00000001;
//		}
//		ERTSplit<R> split = new ERTSplit<R>(attribute, threshold);
//		return split;
//	}
	/**
	 * Return a set of split  for the given attributes.
	 * @param <T> Type of ITrainingPoints used by the Extra Trees.
	 * @param set Set of data points to split.
	 * @param kRandomAttributes attributes for which to create splits.
	 * @return a set of split  for the given attributes.
	 */
	private <T extends ITrainingPoint<R, O>> ArrayList<ERTSplit<R>> getKSplits(Collection<T> set, ArrayList<R> kRandomAttributes, Random fRandom)
	{
		ArrayList<ERTSplit<R>> splits = new ArrayList<ERTSplit<R>>();
		for (R attribute : kRandomAttributes)
		{
			ERTSplit<R> split = createSplit(set, attribute, fRandom);
			splits.add(split);
		}
		return splits;
	}
	
	/**
	 * Return the best split according to the scoring function used by the ERTGrower.
	 * @param <T> Type of ITrainingPoints used by the Extra Trees.
	 * @param set set of points to compute the score with.
	 * @param splits set of splits to compute the score for.
	 * @return the best split according to the scoring function used by the ERTGrower.
	 */
	
	private <T extends ITrainingPoint<R, O>> SplitResult<R, O, T> getBestSplitAndResults(Collection<T> set, ArrayList<R> kRandomAttributes, Random fRandom)
	{
		double maxScore = Double.NEGATIVE_INFINITY;
		SplitResult<R, O, T> ret = null;
		ERTSplit<R> bestSplit = null;
		for (R attribute : kRandomAttributes)
		{
			ERTSplit<R> split = createSplit(set, attribute, fRandom);
			SplitResult<R, O, T> tmp = fScore.getScore(set, split);
			double score = tmp.score;
			if (score > maxScore || bestSplit == null)
			{
				maxScore = score;
				ret = tmp;
				bestSplit = split;
			}
		}
		ret.BestSplit = bestSplit;
		return ret;
	}
}
