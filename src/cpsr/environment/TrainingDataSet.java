/*
 *   Copyright 2012 William Hamilton
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

package cpsr.environment;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;

import java.io.*;
import cpsr.environment.components.ActObSequenceSet;
import cpsr.environment.components.Action;
import cpsr.environment.components.ActionObservation;
import cpsr.environment.components.Observation;
import cpsr.environment.components.ResetActionObservation;

@SuppressWarnings("serial")
public class TrainingDataSet extends DataSet implements Serializable
{
	private static final long serialVersionUID = 8833434998216472206L;
	protected HashSet<Action> validActs;
	protected HashSet<Observation> validObs;
	protected HashSet<ActionObservation> validActObs;
	protected int maxTestLength, maxHistLength;
	protected int runCounter, stepCounter;
	protected int psrRunCounter, psrStepCounter, planBatchCounter;
	public static ActObSequenceSet histories, tests;
	private int currentBatchMaxSize, planRuns;
	private double lastReward;
	private static ResetActionObservation resetAO = new ResetActionObservation();
	// only initialize the tests dictionary and history dictionary once
	{
		if (histories == null && tests == null)
		{
			histories = new ActObSequenceSet();
			tests = new ActObSequenceSet();
		}
	}
	public static ResetActionObservation getResetAO()
	{
		return resetAO;
	}
	public void writeDataInExcel(int epoch) throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter("data" + Integer.toString(epoch) + ".txt"));

		int batchidx = getBatchNumber();
		for (int episodeidx = 0; episodeidx < data.get(batchidx).size(); episodeidx++)
		{
			for (int idx = 0; idx < data.get(batchidx).get(episodeidx).size(); idx++)
			{
				ActionObservation Actob = data.get(batchidx).get(episodeidx).get(idx);
				if (Actob.equals(TrainingDataSet.getResetAO()))
				{
					break;
				}
				int aid = Actob.getAction().getID();
				int oid = Actob.getObservation().getoID();
				Double r = rewards.get(batchidx).get(episodeidx).get(idx);
				writer.write("aid:" + Integer.toString(aid) + ", oid:" + Integer.toString(oid) + ", r:" + Double.toString(r));
				writer.write("\n");
			}
			writer.write("Terminate!");
			writer.write("\n");
		}
		
	    writer.close();
	}
	/**
	 * Explicit default constructor for inheritance.
	 * @param goalSampleRatio TODO
	 */
	
	public TrainingDataSet(int maxTestLength, int maxHistLength)
	{
		super();
		validActObs = new HashSet<ActionObservation>();
		validObs = new HashSet<Observation>();
		validActs = new HashSet<Action>();
		this.maxHistLength = maxHistLength;
		this.maxTestLength = maxTestLength;
		
	}
//	public TrainingDataSet(TrainingDataSet data, double ratio)
//	{
//		super(ratio, data);
//		validActObs = data.validActObs;
//		validObs = data.validObs;
//		validActs = data.validActs;
//		histories = data.histories;
//		tests = data.tests;
//		this.maxHistLength = data.maxHistLength;
//		this.maxTestLength = data.maxTestLength;
//	}
	public TrainingDataSet(TrainingDataSet data, int batch) throws Exception
	{
		super(data, batch);
		validActObs = data.validActObs;
		validObs = data.validObs;
		validActs = data.validActs;
		// histories and tests should be static
//		histories = data.histories;
//		tests = data.tests;
		this.maxHistLength = data.maxHistLength;
		this.maxTestLength = data.maxTestLength;
	}
	/**
	 * Explicit default constructor for inheritance.
	 * @param goalSampleRatio TODO
	 */
	public TrainingDataSet(int maxTestLength, int planRuns, int maxHistLength)
	{
		super();
		validActObs = new HashSet<ActionObservation>();
		validObs = new HashSet<Observation>();
		validActs = new HashSet<Action>();
		//histories and tests should be static
//		histories = new ActObSequenceSet();
//		tests = new ActObSequenceSet();
		this.planRuns = planRuns;
		
		this.maxTestLength = maxTestLength;
		this.maxHistLength = maxHistLength;
	}
	
	public void appendData(TrainingDataSet TrainData)
	{
		this.validActObs.addAll(TrainData.validActObs);
		this.validObs.addAll(TrainData.validObs);
		this.validActs.addAll(TrainData.validActs);
		// Change the tests, histories to static
//		Map<IntSeq, Integer> HMap = TrainData.histories.GetindexMap();
//		Map<IntSeq, Integer> TMap = TrainData.tests.GetindexMap();
//		this.tests.MergeIndexMap(TMap);
//		this.histories.MergeIndexMap(HMap);
		super.appendData(TrainData.data, TrainData.rewards, TrainData.getRunLengths());
	}

	/**
	 * Check and if necessary add a partial sequence to test and history set.
	 * 
	 * @param actobs
	 */
	public void addRunDataForTraining(ArrayList<ActionObservation> actobs)
	{
		if(actobs.size() <= maxHistLength)
		{
			histories.addActObSequence(actobs);
		}
		else
		{
			histories.addActObSequence(actobs.subList(actobs.size()-maxHistLength, actobs.size()));
		}
		
		for(int i = 0; i < actobs.size(); i++)
		{
			if(actobs.size() - i <= maxTestLength)
			{
				tests.addActObSequence(actobs.subList(i, actobs.size()));
			}
		}
		
		ActionObservation mostRecentActOb = actobs.get(actobs.size()-1);
		validActObs.add(mostRecentActOb);
		validActs.add(mostRecentActOb.getAction());
		validObs.add(mostRecentActOb.getObservation());
	}
	
	@Override
	public void addRunData(List<ActionObservation> runActObs, List<Double> runRewards)
	{
		runActObs.add(getResetAO());
		/*
		 * The use of resetActionObservation is to update the last valid actionobservation's C_ao,
		 * Shouldn't include multi-steps tests end at resetAO
		 * Checking it 
		 */
//		for(int i = 0; i < runActObs.size(); i++)
//		{
//			if(runActObs.size() - i <= maxTestLength)
//				tests.addActObSequence(new ArrayList<ActionObservation>(runActObs.subList(i, runActObs.size())));
//		}
		
		tests.addActObSequence(runActObs.subList(runActObs.size()-1, runActObs.size()));
		super.addRunData(runActObs, runRewards);
	}


	/**
	 * Returns next action-observation pair
	 * @return Next action-observation pair
	 */
	public ActionObservation getNextActionObservation()
	{
		ActionObservation actob =  data.get(batchNum).get(runCounter).get(stepCounter);
		stepCounter++;

		if(stepCounter == data.get(batchNum).get(runCounter).size())
		{
			stepCounter = 0;
			runCounter = (runCounter+1)%data.get(batchNum).size();
		}
		return actob;
	}
	
	public boolean IsUpdateRunCounterAndstepCounter(int batchNum, int runCounter, int stepCounter, int offset)
	{
		if(stepCounter == data.get(batchNum).get(runCounter).size() - offset)
		{
			return true;
		}
		return false;
	}

	public int getUpdateRunCounter(int batchNum, int runCounter)
	{
		return (runCounter + 1)%data.get(batchNum).size();
	}

	public ActionObservation getNextActionObservationWithBatchNumrunCounterstepCounter(int batchNum, int runCounter, int stepCounter)
	{
		ActionObservation actob =  data.get(batchNum).get(runCounter).get(stepCounter);
		return actob;
	}


	public double getRewardWithBatchNumrunCounterstepCounter(int batchNum, int runCounter, int stepCounter)
	{
		return rewards.get(batchNum).get(runCounter).get(stepCounter);
	}
	
	/**
	 * Returns next action-observation pair
	 * @return Next action-observation pair
	 */
	public ActionObservation getNextActionObservationForPlanning()
	{
		if(runCounter >= data.get(planBatchCounter).size() || runCounter >= planRuns)
		{
			planBatchCounter++;
			runCounter = 0;
			System.out.print("PlanBatchCounter++!");
		}
		
		ActionObservation actob =  data.get(planBatchCounter).get(runCounter).get(stepCounter);
		lastReward = rewards.get(planBatchCounter).get(runCounter).get(stepCounter);
		stepCounter++;
		if(stepCounter == data.get(planBatchCounter).get(runCounter).size()-1)
		{
			stepCounter = 0;
			int size = data.get(planBatchCounter).size();
			runCounter = (runCounter+1)%(size + 1);
		}
		return actob;
	}
	
	
	
	public void resetData()
	{
		stepCounter = 0;
		runCounter = 0;
		planBatchCounter = 0;
	}
	
	public void setRunCounter(int RunCount)
	{
		this.runCounter = RunCount;
	}
	
	public void setplanBatchCounter(int batch)
	{
		this.planBatchCounter = batch;
	}
	public void resetDatav1()
	{
		stepCounter = 0;
		runCounter = 0;
	}
	
	public void newDataBatch(int maxSize)
	{
		super.newDataBatch(maxSize);
		currentBatchMaxSize = maxSize;
		resetData();
	}
	
	public int getNumberOfRunsInBatch()
	{
		return data.get(batchNum).size();
	}
	
	public void importanceSample(double sampleRatio)
	{
		ArrayList<Integer> goalRuns = new ArrayList<Integer>();
		ArrayList<Integer> badRuns = new ArrayList<Integer>();
		
		for(int i = 0; i < data.get(batchNum).size(); i++)
		{
			if(rewards.get(batchNum).get(i).get(rewards.get(batchNum).get(i).size()-1) > 0.0)
			{
				goalRuns.add(i);
			}
			else
			{
				badRuns.add(i);
			}
		}
		
		List<List<ActionObservation>> sampledBatch = new ArrayList<List<ActionObservation>>();
		List<List<Double>> sampledRewards = new ArrayList<List<Double>>();
		
		Random rando = new Random();
		for(int i = 0; i < currentBatchMaxSize; i++)
		{
			int index;
			
			if(rando.nextDouble() < sampleRatio)
			{
				index = goalRuns.get(rando.nextInt(goalRuns.size()));
			}
			else
			{
				index = badRuns.get(rando.nextInt(badRuns.size()));
			}
			
			sampledBatch.add(data.get(batchNum).get(index));
			sampledRewards.add(rewards.get(batchNum).get(index));
		}
		
		data.set(batchNum, sampledBatch);
		rewards.set(batchNum, sampledRewards);
		
	}

	/**
	 * Returns reward associated with current (i.e. last returned
	 * action observation pair).
	 * 
	 * @return Reward.
	 */
	public double getReward()
	{
		return lastReward;
	}
	
	
	/**
	 * Test whether reset performed on last iterations
	 * 
	 * @return Boolean representing whether reset performed
	 */
	public boolean resetPerformed()
	{
		return stepCounter == 0;
	}

	/**
	 * Get the dimension of observation space
	 * 
	 * @return The size of the observations space
	 */
	public int getNumberObservations()
	{
		return validObs.size();
	}

	/**
	 * Returns an ArrayList of all valid action-observations pairs.
	 * Action-observation pairs are valid if they occur in training set.
	 * 
	 * @return
	 */
	public HashSet<ActionObservation> getValidActionObservationSet()
	{
		return validActObs;
	}

	/**
	 * Returns list of all possible actions in the domain.
	 * @return All possible actions in the domain. 
	 */
	public HashSet<Action> getActionSet()
	{
		return validActs;
	}
	
	public ActObSequenceSet getTests()
	{
		return tests;
	}
	
	public ActObSequenceSet getHistories()
	{
		return histories;
	}
	
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	/**
	 * Returns next action-observation pair
	 * @return Next action-observation pair
	 */
	public ActionObservation getNextActionObservation(int batch)
	{
		ActionObservation actob =  data.get(batch).get(runCounter).get(stepCounter);
		stepCounter++;
		if(stepCounter == data.get(batch).get(runCounter).size())
		{
			stepCounter = 0;
			runCounter = (runCounter+1)%data.get(batch).size();
		}
		return actob;
	}
	
	public int getNumberOfRunsInBatch(int batch)
	{
		return data.get(batch).size();
	}
	
	public int getBatchNumber()
	{
		return this.batchNum;
	}
	

	/////////////////////////////////////////////////////////////////////////////////////////////////////
}
