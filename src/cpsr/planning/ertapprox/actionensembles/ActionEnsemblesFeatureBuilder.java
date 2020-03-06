/*
// *   Copyright 2012 William Hamilton
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
package cpsr.planning.ertapprox.actionensembles;
import cpsr.model.components.FeatureOutcomes;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import cpsr.environment.TrainingDataSet;
import cpsr.environment.components.Action;
import cpsr.environment.components.ActionObservation;
import cpsr.environment.components.Observation;
import cpsr.model.APSR;
import cpsr.model.components.PredictionVector;
import cpsr.planning.exceptions.PSRPlanningException;
import jparfor.Functor;
import jparfor.MultiThreader;
import jparfor.Range;

public class ActionEnsemblesFeatureBuilder {
	TrainingDataSet data;
	APSR psr;
	ArrayList<Double> rewardSet;
	ArrayList<Action> actions;
	boolean actionEnsDone;
	
	/**
	 * Constructs feature builder from an DataSet and PSR.
	 * 
	 * @param data DataSet used to construct features.
	 * @param psr Predictive state representation used to construct features.
	 */
	public ActionEnsemblesFeatureBuilder(TrainingDataSet data, APSR psr)
	{
		this.data = data;
		this.psr = psr;
		//no need to check types match since QFitting does. 
		actions = new ArrayList<Action>(data.getActionSet());
		actionEnsDone = false;
	}
	
	/**
	 * Top level method for building features for action ensemble learning.
	 * 
	 * @param runs Number of runs used in estimation.
	 * @return HashMap mapping action to Hashmap which maps string feature names to
	 * an ordered list of the values they take. 
	 * @throws Exception 
	 */
//	public HashMap<Action, HashMap<String, ArrayList<Double>>> buildFeatures(int runs) throws Exception
//	{
//		HashMap<Action, HashMap<String, ArrayList<Double>>> featureSet = new HashMap<Action, HashMap<String, ArrayList<Double>>>();
//		HashMap<Action, ArrayList<PredictionVector>> predictionVectors = constructDataForActionEnsembles(runs);
//		for(Action act : data.getActionSet())
//		{
//			featureSet.put(act, buildTreeFeaturesForActionEnsemble(act, runs, predictionVectors));
//		}
//		actionEnsDone = true;
//		return featureSet;
//	}
	public ArrayList<PredictionVector> buildFeatures(int runs) throws Exception
	{
//		ArrayList<PredictionVector> predictionVectors = constructDataForActionEnsembles(runs);
		ArrayList<PredictionVector> predictionVectors = constructDataForActionEnsembles_MultiThread();
		actionEnsDone = true;
		return predictionVectors;
	}
	
	/**
	 * Returns list of actions defining order for features
	 */
	public ArrayList<Action> getOrderedListOfActions()
	{
		return actions;
	}
	
	/**
	 * Top level method returns ordered lists of rewards sorted
	 * according to preceding action.
	 * (Note: must call buildActionEnsembleFeatures before calling this
	 * method)
	 * 
	 * @return Ordered lists of rewards sorted according to preceding action
	 */
	public ArrayList<Double> getRewards()
	{
		if(actionEnsDone)
		{
			return rewardSet;
		}
		else
		{
			throw new PSRPlanningException("Must use method buildActionEnsembleFeatures before calling getActionEnsembleRewards");
		}
	}
	
	/**
	 * Builds the input features that will be used with tree in the
	 * case where actions are not included as features.
	 * 
	 * @param Action The action corresponding to this feature set. 
	 * @return The input features.
	 * @throws Exception 
	 */
//	private HashMap<String, ArrayList<Double>> buildTreeFeaturesForActionEnsemble(Action action, int runs, HashMap<Action, ArrayList<PredictionVector>> predictionVectors) throws Exception
//	{
//		HashMap<String, ArrayList<Double>> featuresForAction = intializeFeatureListForActionEnsemble();
//		for(PredictionVector predVec : predictionVectors.get(action))
//		{
//			//TODO: IF THERE ARE PLANNING PROBS IT IS HERE
//			double[] stateFeatures = predVec.getVector().transpose().toArray();
//			for(int i = 0; i < predVec.getVector().getRows(); i++)
//			{
//				featuresForAction.get(Integer.toString(i+1)).add(stateFeatures[i]);
//			}
//		}
//		return featuresForAction;
//	}
//	private ArrayList<double[]> buildTreeFeaturesForActionEnsemble(Action action, int runs, HashMap<Action, ArrayList<PredictionVector>> predictionVectors) throws Exception
//	{
//		ArrayList<double[]> featuresForAction = new ArrayList<double[]>();
//		for(PredictionVector predVec : predictionVectors.get(action))
//		{
//			//TODO: IF THERE ARE PLANNING PROBS IT IS HERE
//			double[] stateFeatures = predVec.getVector().transpose().toArray();
//			featuresForAction.add(stateFeatures);
//		}
//		return featuresForAction;
//	}

	/**
	 * Initializes feature list for action ensemble case
	 * @return The initialized features
	 */
//	private HashMap<String, ArrayList<Double>> intializeFeatureListForActionEnsemble()
//	{
//		HashMap<String, ArrayList<Double>> features = new HashMap<String, ArrayList<Double>>();
//		double[] stateFeatures = psr.getPredictionVector().getVector().transpose().toArray();
//		
//		for(int i = 0; i < stateFeatures.length; i++)
//		{
//			features.put(Integer.toString(i+1), new ArrayList<Double>());
//		}
//		return features;
//	}
	
	
	/**
	 * Constructs the training data in proper format for planning from DataSet
	 * @param runs The number of training runs to use. 
	 * @return A hashmap mapping actions to corresponding list of prediction vectors
	 * @throws Exception 
	 */
	private ArrayList<PredictionVector> constructDataForActionEnsembles(int runs) throws Exception
	{
		psr.resetToStartState();
		actions = new ArrayList<Action>();
		ArrayList<PredictionVector> predictionVectors = new ArrayList<PredictionVector>();
		rewardSet = new ArrayList<Double>();
		int runCount = 0;
		data.resetDatav1();
		int num_action = 0;
		while (runCount < runs)
		{
			stepInActionEnsembleConstruction(predictionVectors);
			num_action++;
			if(checkForReset(predictionVectors))
			{
				runCount++;
			}
				
		}
		System.out.println("The number of actions on FeatureBuilders" + Integer.toString(num_action));
		return predictionVectors;
	}
	
	/*
	 * Merge outcomes into one
	 */
	private FeatureOutcomes MergeOutcomesInList(ArrayList<FeatureOutcomes> BatchOutcomes)
	{
		ArrayList<Action> AListInBatch = new ArrayList<Action>();
		ArrayList<Double> RListInBatch = new ArrayList<Double>();
		ArrayList<PredictionVector> PListInBatch = new ArrayList<PredictionVector>();
		int index = 0;
		while(index < BatchOutcomes.size())
		{
			for (FeatureOutcomes out: BatchOutcomes)
			{
				if (out.index == index)
				{
					AListInBatch.addAll(out.Action);
					RListInBatch.addAll(out.rewardset);
					PListInBatch.addAll(out.PredVectors);
					index++;
					break;
				}
			}
		}
		FeatureOutcomes ret = new FeatureOutcomes();
		ret.Action = AListInBatch;
		ret.PredVectors = PListInBatch;
		ret.rewardset = RListInBatch;
		return ret;
	}
	/*
	 * Generate a sequence of PV for each pair of actions and observations
	 */
	private ArrayList<PredictionVector> constructDataForActionEnsembles_MultiThread() throws Exception
	{
		int BatchNum = data.getBatchNumber();
		psr.resetToStartState();
		ArrayList<FeatureOutcomes> Outcomes = MultiThreader.foreach(new Range(BatchNum + 1), new Functor<Integer, FeatureOutcomes> ()
		{
			@Override
			public FeatureOutcomes function(Integer input) {
				final int Batch = input;
				int ThreadsForOneDataBatch = 4;
				while (data.getNumberOfRunsInBatch(Batch) % ThreadsForOneDataBatch != 0)
				{
					ThreadsForOneDataBatch--;
				}
				final int NumOfCounterPerThread = data.getNumberOfRunsInBatch(Batch) / ThreadsForOneDataBatch;
				ArrayList<FeatureOutcomes> BatchOutcomes = MultiThreader.foreach(new Range(ThreadsForOneDataBatch), new Functor<Integer, FeatureOutcomes>() {
					@Override
					public FeatureOutcomes function(Integer input) {
						ArrayList<Action> AList = new ArrayList<Action>();
						ArrayList<Double> RList = new ArrayList<Double>();
						ArrayList<PredictionVector> PList = new ArrayList<PredictionVector>();
						try {
						APSR psrCopy = psr.clone();
						int RunCounter = input * NumOfCounterPerThread;
						int stepCounter = 0;
						int count = 0;
						while (count < NumOfCounterPerThread)
						{
							ActionObservation ao = data.getNextActionObservationWithBatchNumrunCounterstepCounter(Batch, RunCounter, stepCounter);
							AList.add(Action.GetAction(ao.getAction().getID()));
							RList.add(data.getRewardWithBatchNumrunCounterstepCounter(Batch, RunCounter, stepCounter));
							PredictionVector pv = psrCopy.getPredictionVector();
							PList.add(pv);
							stepCounter++;
							try {
								psrCopy.update(ao);
							} catch (Exception e) {
								e.printStackTrace();
							}
							if (data.IsUpdateRunCounterAndstepCounter(Batch, RunCounter, stepCounter, 1))
							{
								stepCounter = 0;
								RunCounter = data.getUpdateRunCounter(BatchNum, RunCounter);
								count++;
								psrCopy.resetToStartState();
							}
						}
						}catch (Exception e)
						{
							e.printStackTrace();
							throw new RuntimeException(e);
						}
						FeatureOutcomes out = new FeatureOutcomes();
						out.Action = AList;
						out.rewardset = RList;
						out.PredVectors = PList;
						out.index = input;
						return out;
					}
				});
				FeatureOutcomes ret = MergeOutcomesInList(BatchOutcomes);
				ret.index = Batch;
				return ret;
			}
		});
		FeatureOutcomes out = MergeOutcomesInList(Outcomes);
		this.actions = out.Action;
		this.rewardSet = out.rewardset;
		ArrayList<PredictionVector> predictionVectors = out.PredVectors;
		return predictionVectors;
	}
	
	/**
	 * Helper method computes iteration of data construction in
	 * action ensemble case.
	 * 
	 * @param predictionVectors The list of prediction vectors
	 * @param fittedQpsr The PSR.
	 * @throws Exception 
	 */
	private void stepInActionEnsembleConstruction(ArrayList<PredictionVector> predictionVectors) throws Exception
	{
		ActionObservation actob = data.getNextActionObservationForPlanning();
		Action act = actob.getAction();
		actions.add(act);
		rewardSet.add(data.getReward());
		predictionVectors.add(psr.getPredictionVector());
		psr.update(actob);
	}
	

	
	/**
	 * Helper method determines if a run terminated.
	 * If so, true returned and prediction vector reset. 
	 * @param predictionVectors 
	 * 
	 * @return Boolean representing whether reset performed.
	 */
	private boolean checkForReset(ArrayList<PredictionVector> predictionVectors)
	{
		if(data.resetPerformed())
		{
			psr.resetToStartState();
			return true;
		}
		else
		{
			return false;
		}
	}
}
