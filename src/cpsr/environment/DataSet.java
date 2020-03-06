package cpsr.environment;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import Parameter.Param;
import cpsr.environment.components.ActionObservation;

public class DataSet implements Serializable {

	protected int batchNum = -1;
	/**
	 * 
	 */
	private static final long serialVersionUID = 3062257041262864194L;
	/**
	 * @serialField
	 */
	protected List<List<List<ActionObservation>>> data;
	/**
	 * @serialField
	 */
	protected List<List<List<Double>>> rewards;
	private List<Integer> runLengths;
	private static Random rand = new Random(Param.getRandomSeed());
	public DataSet(double ratio, DataSet Traindata)
	{
		batchNum = Traindata.batchNum;
		data = new ArrayList<List<List<ActionObservation>>>();
		rewards = new ArrayList<List<List<Double>>>();
		runLengths = new ArrayList<Integer>();
		for (int batchid = 0; batchid < Traindata.data.size(); batchid++)
		{
			data.add(new ArrayList<List<ActionObservation>>());
			rewards.add(new ArrayList<List<Double>>());
			for (int gameid = 0; gameid < Traindata.data.get(batchid).size(); gameid++)
			{
				if (rand.nextDouble() <= ratio)
				{
					data.get(batchid).add(Traindata.data.get(batchid).get(gameid));
					rewards.get(batchid).add(Traindata.rewards.get(batchid).get(gameid));
					runLengths.add(Traindata.runLengths.get(batchid * Traindata.data.get(batchid).size() + gameid));
				}
			}
		}
	}
	public DataSet(DataSet TrainData, int batch) throws Exception
	{
		batchNum = 0;
		data = new ArrayList<List<List<ActionObservation>>>();
		data.add(TrainData.data.get(batch));
		rewards = new ArrayList<List<List<Double>>>();
		rewards.add(TrainData.rewards.get(batch));
		runLengths = new ArrayList<Integer>();
		int idx = 0;
		for (int i = 0; i < TrainData.data.size(); i++)
		{
			for (int j = 0; j < TrainData.data.get(i).size(); j++)
			{
				int a1 = TrainData.runLengths.get(idx);
				int a2 = TrainData.data.get(i).get(j).size() - 1;
				if (a1 != a2)
				{
					throw new Exception("There are inconsistency on TrainData.");
				}
				if (i == batch)
				{
					runLengths.add(TrainData.runLengths.get(idx));
				}
				idx++;
			}
		}
	}
	
	public DataSet() 
	{
		data = new ArrayList<List<List<ActionObservation>>>();
		rewards = new ArrayList<List<List<Double>>>();
		runLengths = new ArrayList<Integer>();
	}
	
	public List<List<List<ActionObservation>>> get_data()
	{
		return data;
	}
	public List<List<List<Double>>> get_reward()
	{
		return rewards;
	}
	
	public void addRunData(List<ActionObservation> runActObs, List<Double> runRewards)
	{
		data.get(batchNum).add(runActObs);
		rewards.get(batchNum).add(runRewards);
		runLengths.add(runRewards.size());
	}
	/*
	 * Appending the data
	 */
	public void appendData(List<List<List<ActionObservation>>> datanew, List<List<List<Double>>> rewardnew, List<Integer> runLengthsnew)
	{
		if (datanew.size() != 1)
		{
			System.err.println("Strange!");
		}
		this.data.get(batchNum).addAll(datanew.get(0));
		this.rewards.get(batchNum).addAll(rewardnew.get(0));
		this.runLengths.addAll(runLengthsnew);
	}

	public void newDataBatch(int maxSize)
	{
		batchNum++;
		data.add(new ArrayList<List<ActionObservation>>());
		rewards.add(new ArrayList<List<Double>>());
	}


	public List<Integer> getRunLengths()
	{
		return runLengths;
	}
	
	public List<List<Double>> getRewards()
	{
		List<List<Double>> allRewards = new ArrayList<List<Double>>();
	
		for(int i = 0; i < rewards.size(); i++)
		{
			allRewards.addAll(rewards.get(i));
		}
		
		return allRewards;
		
		
	}


}