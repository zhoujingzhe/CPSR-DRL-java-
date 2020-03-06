package cpsr.environment.simulation;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;

import Parameter.Param;
import cpsr.environment.DataSet;
import cpsr.environment.TrainingDataSet;
import cpsr.environment.components.ActObSequenceSet;
import cpsr.environment.components.Action;
import cpsr.environment.components.ActionObservation;
import cpsr.environment.components.IntSeq;
import cpsr.environment.components.Observation;
import cpsr.model.POMDP;
import cpsr.planning.APSRPlanner;
import cpsr.planning.RandomPlanner;

public abstract class ASimulator implements ISimulator 
{

	//use arbitrary maximum run length limit.
	//this simply to prevent infinite loops in simulator
	//and also just practical since runs should not be any longer than this limit
	//if they are to be used with PSRs.
	public int DEFAULT_RUN_LEN_LIMIT = 100000000;
	protected List<String> log_backup;
	protected List<String> log_WayToAct_backup;
	protected List<String> log_validMoves_backup;
	protected List<String> log_WayToAct = new ArrayList<String>();
	protected List<String> log_validMoves = new ArrayList<String>();
	protected String fileName;
	protected int currentcount = 0;
	protected List<String> log = new ArrayList<String>();
	protected long seed;
	protected int maxRunLength;
	protected static Map<Integer, Random> RandoMap=new HashMap<Integer, Random>();;
	
	protected ASimulator(int maxRunLength)
	{
		if(maxRunLength > 1000000)
		{
			throw new IllegalArgumentException("Cannot have max run length greater than: " + DEFAULT_RUN_LEN_LIMIT);
		}
		this.maxRunLength = maxRunLength;
	}

	protected ASimulator()
	{
		this.maxRunLength = DEFAULT_RUN_LEN_LIMIT;
	}	
	
	protected Random Switch_Rando(int seed)
	{
		if (RandoMap.containsKey(seed))
		{
			return RandoMap.get(seed);
		}
		else
		{
			Random rando;
			rando = new Random(Param.getRandomSeed());
			RandoMap.put(seed, rando);
			return rando;
		}
	}
	@Override 
	public DataSet simulateTestRuns(int runs, POMDP planner, Random rando) throws Exception
	{
		int currRun;
		double currReward;
		Observation currObs = null;
		Action currAction = null;

		RandomPlanner randPlanner = null;
		if(planner == null)
			randPlanner = getRandomPlanner(rando);

		DataSet testData = new DataSet();
		//TODO: clean 
		testData.newDataBatch(10000);
		Random randAgent = this.Switch_Rando(Param.getRandomSeed());
		for(currRun = 0; currRun < runs; currRun++)
		{
			currReward = 0.0;
			ArrayList<ActionObservation> runActObs = new ArrayList<ActionObservation>();
			ArrayList<Double> runRewards = new ArrayList<Double>();

			initRun(randAgent);
			if(planner != null)
				planner.reset();
			
			int counter = 0;

			while(!inTerminalState()  && counter < maxRunLength)
			{
				if(planner == null)
				{
					currAction = randPlanner.getAction();
				}
				else
				{
					if (Param.POMDPAction.equals("Action"))
					{
						currAction = Action.GetAction(planner.getAction());
					}
					else if (Param.POMDPAction.equals("Policy"))
					{
						if (currAction == null)
						{
							currAction = Action.GetAction(planner.StartingPolicyGraph());
						}
						else
						{
							currAction = Action.GetAction(planner.getActionByPolicy(currObs.getoID()));
						}
					}
				}
				boolean isopen = executeAction(currAction, randAgent);
				currReward = getCurrentReward();
				currObs = getCurrentObservation(randAgent);
				runActObs.add(ActionObservation.getActionObservation(currAction, currObs));
				runRewards.add(currReward);
				
				if(planner != null)
				{
					planner.updateBelief(ActionObservation.getActionObservation(currAction, currObs));
				}
				counter++;
			}
			testData.addRunData(runActObs, runRewards);
		}
		return testData;
	}
	@Override 
	public DataSet simulateTestRuns(int runs, APSRPlanner planner, Random rando) throws Exception
	{
		int currRun;
		double currReward;
		Observation currObs;
		Action currAction;

		RandomPlanner randPlanner = null;
		if(planner == null)
			randPlanner = getRandomPlanner(rando);

		DataSet testData = new DataSet();
		//TODO: clean 
		testData.newDataBatch(10000);
		Random randAgent = this.Switch_Rando(Param.getRandomSeed());
		for(currRun = 0; currRun < runs; currRun++)
		{
			currReward = 0.0;
			ArrayList<ActionObservation> runActObs = new ArrayList<ActionObservation>();
			ArrayList<Double> runRewards = new ArrayList<Double>();

			initRun(randAgent);
			if(planner != null)
				planner.resetToStartState();
			
			int counter = 0;

			while(!inTerminalState()  && counter < maxRunLength)
			{
				if(planner == null)
				{
					currAction = randPlanner.getAction();
				}
				else
				{
					currAction = planner.getAction();
				}
//				DoubleMatrix actualpv = planner.get_psr().getCoreTestProbability();
//				System.out.println("State:" + actualpv);
//				System.out.println("Action:" + Stand_tiger.Actions[currAction.getID()]);
				boolean isopen = executeAction(currAction, randAgent);
				currReward = getCurrentReward();
				currObs = getCurrentObservation(randAgent);
//				currObs.setMaxID(Param.rewardSize*Param.observationSize-1);
//				currAction.setMaxID(Param.actionSize-1);
				
				runActObs.add(ActionObservation.getActionObservation(currAction, currObs));
				runRewards.add(currReward);
				
				if(planner != null)
				{
					planner.update(ActionObservation.getActionObservation(currAction, currObs));
				}
//				if (isopen)
//				{
//					planner.resetToStartState();
//				}
//				System.out.println("Observation:" + Stand_tiger.Observations[currObs.getoID()]);
//				DoubleMatrix actualpv1 = planner.get_psr().getCoreTestProbability();
//				System.out.println("NewState:" + actualpv1);
				counter++;
			}
			testData.addRunData(runActObs, runRewards);
		}
//		System.out.println("The number of actions:" + Integer.toString(num_action));
		return testData;
	}

	@Override
	public void simulateTrainingRuns(int runs, TrainingDataSet trainData, Random rando) throws Exception 
	{
		simulateTrainingRuns(runs, 0, null, trainData, rando);
	}

	@Override
	public void simulateTrainingRuns(int runs, int initial_seed, APSRPlanner planner, TrainingDataSet trainData, Random rando) throws Exception
	{
		int currRun;
		RandomPlanner randPlanner = getRandomPlanner(rando);
		for(currRun = 0; currRun < runs; currRun++)
		{	
			Random randAction = this.Switch_Rando(currRun + initial_seed);
			ArrayList<ActionObservation> runActObs = new ArrayList<ActionObservation>();
			ArrayList<Double> runRewards = new ArrayList<Double>();
			initRun(randAction);
			if(planner != null)
				planner.resetToStartState();
			Action currAction = null;
			int counter = 0;
			double currReward;
			Observation currObs;
			while(!inTerminalState()  && counter < maxRunLength)
			{
				if(planner != null)
				{
					currAction = planner.getAction();
				}
				else
				{
					currAction = randPlanner.getAction();
				}
				executeAction(currAction, randAction);
				currReward = getCurrentReward();
				currObs = getCurrentObservation(randAction);
//				currObs.setMaxID(getNumberOfObservations()-1);
//				currAction.setMaxID(getNumberOfActions()-1);
				final ActionObservation currentAO = ActionObservation.getActionObservation(currAction, currObs);
				runActObs.add(currentAO);
				runRewards.add(currReward);
				
				trainData.addRunDataForTraining(runActObs);
				if(planner != null)
					planner.update(currentAO);
				counter++;
			}
 			trainData.addRunData(runActObs, runRewards);
			if (runActObs.size() != runRewards.size() + 1)
			{
				System.err.println("The action and reward are not equal");
			}
		}
//		writerDataToExcel(trainData.get_data(), "TrainingData");
	}
		

	@Override
	public void setMaxRunLength(int pMaxRunLength) 
	{
		maxRunLength = pMaxRunLength;
	}
	
	@Override
	public RandomPlanner getRandomPlanner(Random rando)
	{
		return new RandomPlanner(0,0,getNumberOfActions());
	}

	/*
	 * debuging function outputs data
	 */
	protected void writerDataToExcel(List<List<List<ActionObservation>>> data, String filename)
	{
		HSSFWorkbook workbook = new HSSFWorkbook();
        HSSFSheet sheet = workbook.createSheet(filename);
        int rownum=0;
		for (int Batchidx=0; Batchidx < data.size(); Batchidx++)
		{
			for(int runidx=0; runidx < data.get(Batchidx).size(); runidx++)
			{
				Row row = sheet.createRow(rownum++);
	            List<ActionObservation> actoblist = data.get(Batchidx).get(runidx);
	            IntSeq ID = ActObSequenceSet.computeID(actoblist);
	            int cellnum = 0;
	            Cell cell = row.createCell(cellnum++);
	            Cell cell2 = row.createCell(cellnum++);
	            cell.setCellValue(actoblist.toString());
	            cell2.setCellValue(ID.toString());
			}
		}
					  
        try {
            FileOutputStream out
                    = new FileOutputStream(new File(filename + ".xls"));
            workbook.write(out);
            out.close();
            System.out.println("Excel written successfully..");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
	}
	protected abstract int[][] initRun(Random rando);

	protected abstract int getNumberOfActions();
	
	protected abstract int getNumberOfObservations();
	
	protected abstract boolean inTerminalState();
	
	protected abstract boolean executeAction(Action act, Random rando);

	protected abstract double getCurrentReward();
	
	protected abstract Observation getCurrentObservation(Random rando);
}
