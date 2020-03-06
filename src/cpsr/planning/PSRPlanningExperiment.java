package cpsr.planning;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;

import org.jblas.DoubleMatrix;

///////////////////////////////////////////////////////////////////////
import com.google.gson.Gson;

import Parameter.Param;
import afest.datastructures.tree.decision.erts.ERTTrainingPoint;
import cpsr.environment.DataSet;
import cpsr.environment.TrainingDataSet;
import cpsr.environment.components.Action;
import cpsr.environment.components.ActionObservation;
import cpsr.environment.components.Observation;
import cpsr.environment.simulation.ISimulator;
import cpsr.environment.simulation.domains.HarderPacMan;
import cpsr.environment.simulation.domains.Maze;
import cpsr.environment.simulation.domains.NoisyPacMan;
import cpsr.environment.simulation.domains.PacMan;
import cpsr.environment.simulation.domains.Stand_tiger;
import cpsr.environment.simulation.domains.Stand_tiger_Test;
import cpsr.environment.simulation.domains.Tiger95;
import cpsr.environment.simulation.domains.niceEnv;
import cpsr.environment.simulation.domains.shuttle;
import cpsr.model.APSR;
import cpsr.model.CPSR.ProjType;
import cpsr.model.HashedCPSR;
import cpsr.model.MemEffCPSR;
import cpsr.model.MemEffCPSR_NotCompressingHistory;
import cpsr.model.MemorylessState;
import cpsr.model.POMDP;
import cpsr.model.PSR;
import cpsr.model.Predictor;
import cpsr.model.TPSR;
import cpsr.model.components.PredictionVector;
import cpsr.planning.ertapprox.actionensembles.ActionERTQPlanner;
import cpsr.planning.ertapprox.actionensembles.ActionERTQPlannerDRaPS;
import cpsr.stats.Likelihood;
import cpsr.stats.PSRObserver;
import cpsr.stats.PSRPlanningExperimentStatPublisher;
import jparfor.Functor;
import jparfor.MultiThreader;
import jparfor.Range;

///////////////////////////////////////////////////////////////////////
public class PSRPlanningExperiment 
{

	public static final String DEF_ENSEMBLE_TYPE = "action",  DEF_LEAF_SIZE = "5", DEF_NUM_TREES="30", 
			DEF_TREE_BUILDING_ITER="50", DEF_EPSILON = "0.1", DEF_TREES_PER_ENS = "30", DEF_DISCOUNT_FACTOR = "0.9",
			DEF_PLANNING_ITERATIONS="30", DEF_TEST_RUNS="10000", DEF_MIN_SING_VAL = "0.000000001", DEF_MODEL_LEARN_TYPE="update",
			DEF_PROJ_TYPE="Spherical", DEF_RAND_START="false", DEF_INIT_RUNS="1000";

	private ISimulator simulator;

	private int svdDim, projDim, policyIter, runsPerIter, maxTestLen, maxRunLength,
	numTreeSplits, leafSize, numTrees, treeIters, maxHistLen, testRuns, initRuns;

	private double epsilon, discount, minSingVal;

	private String planningType, modelLearnType;

	boolean memoryless, randStart;

	private ProjType projType;

	private Properties psrProperties, plannerProperties;

	private DataSet testResults;

	private List<List<List<ActionObservation>>> trainRunResults;
	private List<List<List<Double>>> trainRunRewards;
	private List<Double> modelBuildTimes, policyConstructionTimes;

	private PSRPlanningExperimentStatPublisher publisher;
	private PSRObserver psrObs;

	private boolean histCompress, hashed;

	private int maxBatchSize;

	private double sampleRatio;

	private List<List<Double>> randRewards;

	private int planIters;

	private static int seed = 1234567;

	/**
	 * Constructs a planning experiment.
	 * 
	 * @param pPSRConfigFile 
	 * @param pPlanningConfigFile
	 * @param pSimulator
	 */
	public PSRPlanningExperiment(String pPSRConfigFile, String pPlanningConfigFile, ISimulator pSimulator)
	{
		psrProperties = new Properties();
		plannerProperties = new Properties();

		simulator = pSimulator;


		try
		{
			psrProperties.load(new FileReader(pPSRConfigFile));
			plannerProperties.load(new FileReader(pPlanningConfigFile));

			//getting PSR parameters.
			memoryless = Boolean.parseBoolean(psrProperties.getProperty("Memoryless", "false"));
			svdDim = Integer.parseInt(psrProperties.getProperty("SVD_Dimension", "-1"));
			projDim = Integer.parseInt(psrProperties.getProperty("Projection_Dimension", "-1"));
			maxTestLen = Integer.parseInt(psrProperties.getProperty("Max_Test_Length", "-1"));
			maxHistLen = Integer.parseInt(psrProperties.getProperty("Max_History_Length", "-1"));
			minSingVal = Double.parseDouble(psrProperties.getProperty("Min_Singular_Val", DEF_MIN_SING_VAL));
			projType = ProjType.valueOf(psrProperties.getProperty("Projection_Type", DEF_PROJ_TYPE));
			randStart = Boolean.parseBoolean(psrProperties.getProperty("Rand_Start", DEF_RAND_START));
			histCompress = Boolean.parseBoolean(psrProperties.getProperty("Hist_Compress", "false"));
			hashed = Boolean.parseBoolean(psrProperties.getProperty("Hashed", "false"));
			
			
			//getting planning parameters
			planningType = plannerProperties.getProperty("Ensemble_Type", DEF_ENSEMBLE_TYPE);
			epsilon = Double.parseDouble(plannerProperties.getProperty("Epsilon", DEF_EPSILON));
			numTreeSplits = Integer.parseInt(plannerProperties.getProperty("Num_Tree_Splits", "-1"));
			runsPerIter = Integer.parseInt(plannerProperties.getProperty("Runs_Per_Iteration"));
			leafSize = Integer.parseInt(plannerProperties.getProperty("Leaf_Size", DEF_LEAF_SIZE));
			numTrees = Integer.parseInt(plannerProperties.getProperty("Trees_Per_Ensemble", DEF_TREES_PER_ENS));
			treeIters = Integer.parseInt(plannerProperties.getProperty("Tree_Building_Iterations", DEF_TREE_BUILDING_ITER));
			policyIter = Integer.parseInt(plannerProperties.getProperty("Planning_Iterations", DEF_PLANNING_ITERATIONS));
			testRuns = Integer.parseInt(plannerProperties.getProperty("Test_Runs", DEF_TEST_RUNS));
			discount = Double.parseDouble(plannerProperties.getProperty("Discount_Factor", DEF_DISCOUNT_FACTOR));
			maxRunLength = Integer.parseInt(plannerProperties.getProperty("Max_Run_Length", "-1"));
			modelLearnType = plannerProperties.getProperty("Model_Learn_Type", DEF_MODEL_LEARN_TYPE);
			initRuns =  Integer.parseInt(plannerProperties.getProperty("Init_Runs", DEF_INIT_RUNS));
			maxBatchSize = Integer.parseInt(plannerProperties.getProperty("Max_Batch_Size", "1000"));
			sampleRatio = Double.parseDouble(plannerProperties.getProperty("Sample_Ratio", "-1.0"));
			planIters = Integer.parseInt(plannerProperties.getProperty("Plan_Iters", Integer.toString(runsPerIter)));
			Param.trees = numTrees;
			Param.SVD_DIM = svdDim;
			if (maxHistLen == -1) maxHistLen = Integer.MAX_VALUE;
			if (maxTestLen == -1) maxTestLen = Integer.MAX_VALUE;
			if(maxRunLength == -1) maxRunLength = Integer.MAX_VALUE;

		}
		catch(IOException ex)
		{
			ex.printStackTrace();
		}

		if(numTreeSplits == -1 || runsPerIter == 0 || maxTestLen == -1 || maxHistLen == -1 || svdDim == -1)
		{
			throw new IllegalArgumentException("Missing required parameter");
		}
	}

	/**
	 * Runs an experiment and returns the DataSet of test results. 
	 * 
	 * @return Test results;
	 * @throws Exception 
	 */
	public DataSet runExperiment(int gameidx, String modelName, String PSRPath) throws Exception
	{
		APSR psrModel = null;
		APSRPlanner planner = null;
		trainRunResults = new ArrayList<List<List<ActionObservation>>>();
		trainRunRewards = new ArrayList<List<List<Double>>>();
		modelBuildTimes = new ArrayList<Double>();
		policyConstructionTimes = new ArrayList<Double>();
		psrObs = new PSRObserver();
		publisher = new PSRPlanningExperimentStatPublisher(this, simulator.getName(), psrObs, psrProperties, plannerProperties);
		simulator.setMaxRunLength(maxRunLength);
		TrainingDataSet trainData = new TrainingDataSet(maxTestLen, planIters, maxHistLen);
		TrainingDataSet trainDataForStandTiger = new TrainingDataSet(maxTestLen, planIters, maxHistLen);
	
		///////////////////////////////////////////////////////////////////////////
		// loading previous policy and psr model
		if (Param.loading)
		{
			System.out.println("Loading existed policy and psr model");
			FileInputStream fileIn = new FileInputStream("Game0FittedQ_PolicyIteration1.ser");
	        ObjectInputStream in = new ObjectInputStream(fileIn);
	        planner = (APSRPlanner) in.readObject();
	        in.close();
	        fileIn.close();
//	        psrModel = planner.get_psr();
		}
		///////////////////////////////////////////////////////////////////////////
		
		if(sampleRatio != -1.0)
			trainData.importanceSample(sampleRatio);
//		FileInputStream fileIn = new FileInputStream("data.ser");
//        ObjectInputStream in = new ObjectInputStream(fileIn);
//        trainData = (TrainingDataSet) in.readObject();
//        in.close();
//        fileIn.close();
		// initial the PSR model
		if(memoryless)
		{
			psrModel = new MemorylessState(trainData);
		}
		else if (psrModel == null)
		{
//			if (simulator.getName().equals("Stand_tiger"))
//			{
//				psrModel = initPSRModel(modelName, trainDataForStandTiger);
//			}
//			else
//			{
				psrModel = initPSRModel(modelName, trainData);
//			}
			
			psrModel.addPSRObserver(psrObs);
			if (modelName.equals("PSR"))
			{
				psrModel.loadingExistPSR(PSRPath);
				psrModel.build(svdDim, Param.seed);
			}
		}
//		 if having loaded a PSR model, directly using EpsilonGreedy policy to simulate, otherwise, random exploration
		trainData.newDataBatch(maxBatchSize);
		trainDataForStandTiger.newDataBatch(maxBatchSize);
		if (planner == null)
		{
//			if (simulator.getName().equals("Stand_tiger"))
//			{
//				simulator.simulateTrainingRuns(initRuns, trainDataForStandTiger, new Random(Param.getRandomSeed()));
//				simulator.simulateTrainingRuns(runsPerIter, 0,  null, trainData, new Random(Param.getRandomSeed()));
//			}
//			else
//			{
				simulator.simulateTrainingRuns(initRuns, trainData, new Random(Param.getRandomSeed()));
//			}
		}
		else
		{
			simulator.simulateTrainingRuns(initRuns, 0, new EpsilonGreedyPlanner(simulator.getRandomPlanner(new Random(0)), planner, epsilon, new Random(0)), trainData, new Random(0));
		}
		randRewards = trainData.getRewards();
//		FileOutputStream fileout = new FileOutputStream("data.ser");
//        ObjectOutputStream in = new ObjectOutputStream(fileout);
//        in.writeObject(trainData);
//        in.close();
//        fileout.close();
		// Data-Expansion Iterations
		for(int i = 0; i < policyIter; i++)
		{			
			System.out.println("Starting " + Integer.toString(i + 1) + " Policy Iteration!");
			long startBuildTime = System.currentTimeMillis();
			// build and update the models if it is not PurePSR model
			if (!modelName.equals("PSR"))
			{
				if(modelLearnType.equals("update") && i != 0)
				{
					if (Param.Update.equals("update"))
					{
						// incremental update
						psrModel.update();
					}
					else if (Param.Update.equals("rebuild"))
					{
						//rebuild the CPSR model based on all of data
//						if (!simulator.getName().equals("Stand_tiger"))
//						{
							psrModel.build(svdDim, Param.seed);
//						}
					}
					else
					{
						throw new Exception("Updating the PSR model has problems!");
					}
				}
				else
				{	
					psrModel.build(svdDim, Param.seed);
				}
			}
			
			if (planner == null && Param.planningType.equals("action"))
			{
				// create Policy learner
				System.out.println("actionERTQPlanner!");
				planner = new ActionERTQPlanner(psrModel);
			}
			else if (planner == null && Param.planningType.equals("DRaPS"))
			{
				System.out.println("actionDRaPSERTQPlanner!");
				planner = new ActionERTQPlannerDRaPS(psrModel);
			}
			else if (planner == null)
			{
				throw new Exception("The planningType is unknown!");
			}
//			/////////////////////////////////////////////////////////////////////////////////////////////////////
//			/*
//			 * output the likelihoods of all one step tests.
//			 */
			psrModel.resetToStartState();
			Likelihood like = new Likelihood(psrModel, trainData, (new Predictor(psrModel))); 
			Object[] listao = trainData.getValidActionObservationSet().toArray();
			List<ActionObservation> list_ao = new ArrayList<ActionObservation>();
			for (Object ao:listao)
			{
				list_ao.add((ActionObservation) ao);
			}
			try {
				/*
				 * Maze
				 */
				if (simulator.getName().equals("Maze"))
				{
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_null", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(0), Observation.GetObservation(0, -1)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "a0o0", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(2), Observation.GetObservation(2, -1)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "a0o0a2o2", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(2), Observation.GetObservation(2, -1)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "a0o0a2o2a2o2", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(2), Observation.GetObservation(4, -1)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "a0o0a2o2a2o2a2o4", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(4), Observation.GetObservation(0, -1)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "a0o0a2o2a2o2a2o4a4o0", simulator.getName());
				}
				/*
				 * shuttle
				 */
				if (simulator.getName().equals("shuttle"))
				{
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_null", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(1), Observation.GetObservation(3, -1)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "a1o3", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(1), Observation.GetObservation(0, -1)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "a1o3a1o0", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(1), Observation.GetObservation(0, -1)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "a1o3a1o0a1o0", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(0), Observation.GetObservation(3, -1)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "a1o3a1o0a1o0a0o3", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(2), Observation.GetObservation(3, -1)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "a1o3a1o0a1o0a0o3a2o3", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(2), Observation.GetObservation(4, -1)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "a1o3a1o0a1o0a0o3a2o3a2o4", simulator.getName());
				}
				
				/*
				 * niceEnv
				 */
				if (simulator.getName().equals("niceEnv"))
				{
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_null", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(1), Observation.GetObservation(2, 2)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a1o2", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(2), Observation.GetObservation(3, 2)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a1o2a2o3", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(2), Observation.GetObservation(4, 2)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a1o2a2o3a2o4", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(0), Observation.GetObservation(3, 2)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a1o2a2o3a2o4a0o3", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(0), Observation.GetObservation(2, 2)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a1o2a2o3a2o4a0o3a0o2", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(0), Observation.GetObservation(2, 2)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a1o2a2o3a2o4a0o3a0o2a0o2", simulator.getName());
					psrModel.resetToStartState();
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(0), Observation.GetObservation(1, 2)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a0o1", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(0), Observation.GetObservation(0, 2)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a0o1a0o0", simulator.getName());
				}
				
				/*
				 * Tiger95
				 */
				if (simulator.getName().equals("Tiger95"))
				{
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_null", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(2), Observation.GetObservation(0, 2)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a2o0", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(2), Observation.GetObservation(0, 2)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a2o0a2o0", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(2), Observation.GetObservation(1, 2)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a2o0a2o0a2o1", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(2), Observation.GetObservation(1, 2)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a2o0a2o0a2o1a2o1", simulator.getName());
					psrModel.resetToStartState();
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(2), Observation.GetObservation(1, 2)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a2o1", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(2), Observation.GetObservation(1, 2)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a2o1a2o1", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(2), Observation.GetObservation(0, 2)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a2o1a2o1a2o0", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(2), Observation.GetObservation(0, 2)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a2o1a2o1a2o0a2o0", simulator.getName());
				}

				/*
				 * Stand Tiger
				 */
				if (simulator.getName().equals("Stand_tiger"))
				{
			
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_null", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(3), Observation.GetObservation(0, 0)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a3o0r0", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(3), Observation.GetObservation(0, 0)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a3o0r0a3o0r0", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(4), Observation.GetObservation(3, 0)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a3o0r0a3o0r0a4o3r0", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(2), Observation.GetObservation(2, 2)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a3o0r0a3o0r0a4o3r0a2o2r2", simulator.getName());
					psrModel.resetToStartState();
					
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(3), Observation.GetObservation(1, 0)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a3o1r0", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(3), Observation.GetObservation(1, 0)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a3o1r0a3o1r0", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(4), Observation.GetObservation(4, 0)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a3o1r0a3o1r0a4o4r0", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(0), Observation.GetObservation(1, 1)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a3o1r0a3o1r0a4o4r0a1o1r1", simulator.getName());
					psrModel.resetToStartState();
					
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(3), Observation.GetObservation(2, 0)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a3o2r0", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(3), Observation.GetObservation(2, 0)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a3o2r0a3o2r0", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(4), Observation.GetObservation(5, 0)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a3o2r0a3o2r0a4o5r0", simulator.getName());
					psrModel.update(ActionObservation.getActionObservation(Action.GetAction(2), Observation.GetObservation(0, 1)));
					like.PrintOutLikelihoodsOfAllao(list_ao, i, "likelihood_a3o2r0a3o2r0a4o5r0a2o0r1", simulator.getName());
					psrModel.resetToStartState();
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
			psrModel.resetToStartState();
//			if (i > 0)
//			{
//			trainData.writeDataInExcel(i);
//			}

//			///////////////////////////////////////////////////////////////////////////////////////////
			long endBuildTime = System.currentTimeMillis();
			modelBuildTimes.add((double)(endBuildTime-startBuildTime)/1000);
			System.out.println("Constructing the CPSR costs " + Double.toString((double)(endBuildTime-startBuildTime)/1000));
			long startPolicyTime = System.currentTimeMillis();
			// remove useless tree blocks.
			// learning the policy based on all simulated trajectories
//			if(i == 0)
//			{
//				planner.learnQFunction(trainData, planIters, treeIters, numTreeSplits, leafSize,
//						numTrees, discount, i);
//			}
//			else
//			{
//				planner.learnQFunction(trainData, planIters+i*Math.min(runsPerIter, planIters), treeIters, numTreeSplits, leafSize,
//						numTrees, discount, i);
//			}
			long endPolicyTime = System.currentTimeMillis();
			policyConstructionTimes.add((double)(endPolicyTime-startPolicyTime)/1000);
			System.out.println("Policy Construction costs " + Double.toString((double)(endPolicyTime-startPolicyTime)/1000));
			System.out.println("Finished iteration: " + (i+1));
			// continuously re-simulate the environment
			if(i != policyIter-1)
			{
				trainData.newDataBatch(maxBatchSize);
				simulator.simulateTrainingRuns(initRuns, trainData, new Random(Param.getRandomSeed()));
//				APSRPlanner plannerThread = planner;
//				String game = simulator.getName();
//				ArrayList<MultiSimulatedResult> TrainDataList = MultiThreader.foreach(new Range(runsPerIter/100), new Functor<Integer, MultiSimulatedResult>() {
//					@Override
//					public MultiSimulatedResult function(Integer input) {
//						// TODO Auto-generated method stub
//						MultiSimulatedResult out= null;
//						try {
//						ISimulator Pac1 = null;
//						if (game.equals("Tiger95"))
//						{
//							Pac1 = new Tiger95(maxRunLength);
//						}
//						else if (game.equals("PacMan"))
//						{
//							Pac1 = new PacMan(maxRunLength);
//						}
//						else if (game.equals("NPacMan"))
//						{
//							Pac1 = new NoisyPacMan(maxRunLength);
//						}
//						else if (game.equals("HPacMan"))
//						{
//							Pac1 = new HarderPacMan(maxRunLength);
//						}
//						else if (game.equals("Stand_tiger"))
//						{
//							Pac1 = new Stand_tiger(maxRunLength);
//						}
//						else if (game.equals("niceEnv"))
//						{
//							Pac1 = new niceEnv(maxRunLength);
//						}
//						else if (game.equals("shuttle"))
//						{
//							Pac1 = new shuttle(maxRunLength);
//						}
//						else if (game.equals("Maze"))
//						{
//							Pac1 = new Maze(maxRunLength);
//						}
//						else if (game.equals("Stand_tiger_Test"))
//						{
//							Pac1 = new Stand_tiger_Test(maxRunLength);
//						}
//						else
//						{
//							System.err.println("No such games");
//						}
//						TrainingDataSet traindata1 = new TrainingDataSet(maxTestLen, planIters, maxHistLen);
//						traindata1.newDataBatch(maxBatchSize);
//						APSRPlanner Planner1 = plannerThread.clone();
//						Planner1.resetToStartState();
//						try {
//							Pac1.simulateTrainingRuns(100, 100*input,  new EpsilonGreedyPlanner(Pac1.getRandomPlanner(new Random(Param.getRandomSeed())), Planner1, epsilon, new Random(Param.getRandomSeed())), traindata1, new Random(Param.getRandomSeed()));
//							out = new MultiSimulatedResult();
//							out.TList = traindata1;
//							out.index = input;
//						} catch (Exception e) {
//							// TODO Auto-generated catch block
//							e.printStackTrace();
//						}
//						}catch (Exception e)
//						{
//							e.printStackTrace();
//							throw new RuntimeException(e);
//						} 
//						return out;
//					}
//				});
//				MergingDataSet(TrainDataList, trainData);
			}
//			testResults = simulator.simulateTestRuns(testRuns, planner, new Random(3));
//			this.publishResults("Evaluate/" + simulator.getName() + "EvalResultGId" + Integer.toString(gameidx) + "PI" + Integer.toString(i+1));
//			testResults = null;
//			//////////////////////////////////////////////////////////////////
//			FileOutputStream fileOut = new FileOutputStream("data.ser");
//			ObjectOutputStream out = new ObjectOutputStream(fileOut);
//			out.writeObject(trainData);
//			out.close();
//			fileOut.close();
			// saving planner
//			if (i%10 == 0)
//			{
//				APSRPlanner plannerclone = planner.clone();
//				if (Param.C51)
//				{
//					FileOutputStream fileOut = new FileOutputStream("Game"+Integer.toString(gameidx)+"C51_PolicyIteration"+ Integer.toString(i) +".ser");
//					ObjectOutputStream out = new ObjectOutputStream(fileOut);
//					out.writeObject(plannerclone);
//					out.close();
//					fileOut.close();
//				}
//				else
//				{
//					FileOutputStream fileOut = new FileOutputStream("Game"+Integer.toString(gameidx)+"FittedQ_PolicyIteration"+ Integer.toString(i) +".ser");
//					ObjectOutputStream out = new ObjectOutputStream(fileOut);
//					out.writeObject(planner);
//					out.close();
//					fileOut.close();
//				}
//			}
		}
		//////////////////////////////////////////////////////////////////
//		/// ZJZ modified
		PredictionVector pv = psrModel.getPredictionVector();
		Map<ActionObservation, DoubleMatrix> aoMats = psrModel.getAOMats();
		Gson gson = new Gson();
		String json_pv = gson.toJson(pv);
		try {
			String json_aoMats = gson.toJson(aoMats);
			System.out.println(aoMats.size());
	        //Write JSON file
	        try (FileWriter file = new FileWriter("predictive_vector.json")) {
	            file.write(json_pv);
	            file.flush();
	            file.close();
	        } catch (IOException e) {
	            e.printStackTrace();
	        }
	        try (FileWriter file = new FileWriter("aoMats.json")) {
	            file.write(json_aoMats);
	            file.flush();
	            file.close();
	        } catch (IOException e) {
	            e.printStackTrace();
	        }
		} catch (IllegalArgumentException i1){
			for (Map.Entry<ActionObservation, DoubleMatrix> entry : aoMats.entrySet())
			{
				ActionObservation ao = entry.getKey();
				DoubleMatrix mat = entry.getValue();
				System.err.println("Error");
				System.out.println("ao:" + ao);
				System.out.println(mat.toString());
			}
		}
		//////////////////////////////////////////////////////////////////
		return testResults;
	}
	
	/*
	 * Combining a list of trainingData into one
	 */
	private void MergingDataSet(List<MultiSimulatedResult> TrainDataList, TrainingDataSet TrainData) {
		int index = 0;
		while(index < TrainDataList.size())
		{
			for (MultiSimulatedResult TrainData1:TrainDataList)
			{
				if (TrainData1.index == index)
				{
					TrainData.appendData(TrainData1.TList);
					index++;
					break;
				}
			}
		}
	}
//	
	// loading existing PSR model and run experiment
	
	public DataSet Eval(String path) throws Exception
	{
		APSR psrModel = null;
		APSRPlanner planner = null;
		trainRunResults = new ArrayList<List<List<ActionObservation>>>();
		trainRunRewards = new ArrayList<List<List<Double>>>();
		modelBuildTimes = new ArrayList<Double>();
		policyConstructionTimes = new ArrayList<Double>();
		psrObs = new PSRObserver();
		publisher = new PSRPlanningExperimentStatPublisher(this, simulator.getName(), psrObs, psrProperties, plannerProperties);

		simulator.setMaxRunLength(maxRunLength);

		TrainingDataSet trainData = new TrainingDataSet(maxTestLen, planIters, maxHistLen);
		
		///////////////////////////////////////////////////////////////////////////
		// loading previous policy and psr model
		if (true)
		{
			System.out.println("Loading existed policy and psr model");
			FileInputStream fileIn = new FileInputStream(path);
	        ObjectInputStream in = new ObjectInputStream(fileIn);
	        planner = (APSRPlanner) in.readObject();
	        in.close();
	        fileIn.close();
	        psrModel = planner.get_psr();
		}
		trainData.newDataBatch(maxBatchSize);
		simulator.simulateTrainingRuns(initRuns, trainData, new Random(Param.getRandomSeed()));
		randRewards = trainData.getRewards();
//		psrModel.resetToStartState();
//		Likelihood like = new Likelihood(psrModel, trainData, (new Predictor(psrModel))); 
//		Object[] listao = trainData.getValidActionObservationSet().toArray();
//		List<ActionObservation> list_ao = new ArrayList<ActionObservation>();
//		for (Object ao:listao)
//		{
//			list_ao.add((ActionObservation) ao);
//		}
//		try {
//			like.PrintOutLikelihoodsOfAllao(list_ao, 22, "likelihood_nullhistory", simulator.getName());
////			psrModel.update(ActionObservation.getActionObservation(Action.GetAction(3), Observation.GetObservation(2, 3)));
////			Likelihood like2 = new Likelihood(psrModel, trainData, (new Predictor(psrModel))); 
////			like2.PrintOutLikelihoodsOfAllao(list_ao, 22, "likelihood_a3o2r3", simulator.getName());
////			psrModel.update(ActionObservation.getActionObservation(Action.GetAction(4), Observation.GetObservation(5, 3)));
////			Likelihood like3 = new Likelihood(psrModel, trainData, (new Predictor(psrModel))); 
////			like3.PrintOutLikelihoodsOfAllao(list_ao, 22, "likelihood_a4o5r3", simulator.getName());
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
//		psrModel.resetToStartState();
//		randRewards = trainData.getRewards();
		testResults = simulator.simulateTestRuns(testRuns, planner, new Random(Param.getRandomSeed()));		
		return testResults;
	}
	
	public DataSet EvalPOMDP() throws Exception
	{
		trainRunResults = new ArrayList<List<List<ActionObservation>>>();
		trainRunRewards = new ArrayList<List<List<Double>>>();
		modelBuildTimes = new ArrayList<Double>();
		policyConstructionTimes = new ArrayList<Double>();
		psrObs = new PSRObserver();
		publisher = new PSRPlanningExperimentStatPublisher(this, simulator.getName(), psrObs, psrProperties, plannerProperties);

		simulator.setMaxRunLength(maxRunLength);

		TrainingDataSet trainData = new TrainingDataSet(maxTestLen, planIters, maxHistLen);
		
		///////////////////////////////////////////////////////////////////////////
		POMDP model = simulator.generatePOMDP();
		trainData.newDataBatch(maxBatchSize);
		simulator.simulateTrainingRuns(initRuns, trainData, new Random(Param.getRandomSeed()));
		randRewards = trainData.getRewards();
		testResults = simulator.simulateTestRuns(testRuns, model, new Random(Param.getRandomSeed()));		
		return testResults;
	}
	
	public static String getDefEnsembleType() {
		return DEF_ENSEMBLE_TYPE;
	}

	public static String getDefLeafSize() {
		return DEF_LEAF_SIZE;
	}

	public static String getDefNumTrees() {
		return DEF_NUM_TREES;
	}

	public static String getDefTreeBuildingIter() {
		return DEF_TREE_BUILDING_ITER;
	}

	public static String getDefEpsilon() {
		return DEF_EPSILON;
	}

	public static String getDefTreesPerEns() {
		return DEF_TREES_PER_ENS;
	}

	public static String getDefDiscountFactor() {
		return DEF_DISCOUNT_FACTOR;
	}

	public static String getDefPlanningIterations() {
		return DEF_PLANNING_ITERATIONS;
	}

	public static String getDefTestRuns() {
		return DEF_TEST_RUNS;
	}

	public static String getDefMinSingVal() {
		return DEF_MIN_SING_VAL;
	}

	public static String getDefModelLearnType() {
		return DEF_MODEL_LEARN_TYPE;
	}

	public static String getDefProjType() {
		return DEF_PROJ_TYPE;
	}

	public static String getDefRandStart() {
		return DEF_RAND_START;
	}

	public static String getDefInitRuns() {
		return DEF_INIT_RUNS;
	}

	public ISimulator getSimulator() {
		return simulator;
	}

	public int getInitRuns() {
		return initRuns;
	}

	public boolean isMemoryless() {
		return memoryless;
	}

	public boolean isRandStart() {
		return randStart;
	}

	public PSRPlanningExperimentStatPublisher getPublisher() {
		return publisher;
	}

	public PSRObserver getPsrObs() {
		return psrObs;
	}

	public boolean isHistCompress() {
		return histCompress;
	}

	public boolean isHashed() {
		return hashed;
	}

	public int getMaxBatchSize() {
		return maxBatchSize;
	}

	public double getSampleRatio() {
		return sampleRatio;
	}

	public int getPlanIters() {
		return planIters;
	}

	public static int getSeed() {
		return seed;
	}

	public void publishResults(String resultsDir)
	{
		publisher.publishResults(resultsDir);
	}

	private APSR initPSRModel(String psrType, TrainingDataSet trainData)
	{
		APSR psr = null;
		if(psrType.equals("CPSR"))
		{
			if(hashed)
			{
				if(maxHistLen != -1)
				{
					psr = new HashedCPSR(trainData, minSingVal, svdDim, maxHistLen, projDim, histCompress, maxTestLen);
				}
				else
				{
					psr = new HashedCPSR(trainData, minSingVal, svdDim, projDim, histCompress);
				}
			}
			else
			{
				if(maxHistLen != -1)
				{
					psr = new MemEffCPSR(trainData, minSingVal, svdDim, projDim, maxHistLen, projType, randStart, maxTestLen);
				}
				else
				{
					psr = new MemEffCPSR(trainData, minSingVal, svdDim, projDim, projType,  randStart);
				}
			}
		}
		else if (psrType.equals("TPSR"))
		{
			if(maxHistLen != -1)
			{
				psr = new TPSR(trainData, minSingVal, svdDim, maxHistLen, maxTestLen);
			}
			else
			{
				psr = new TPSR(trainData, minSingVal, svdDim);
			}
		}
		else if (psrType.equals("PSR"))
		{
			psr = new PSR(trainData);
		}
		return psr;
	}

	public int getSvdDim() {
		return svdDim;
	}

	public int getProjDim() {
		return projDim;
	}

	public ProjType getProjType(){
		return projType;
	}

	public int getPolicyIter() {
		return policyIter;
	}

	public int getRunsPerIter() {
		return runsPerIter;
	}

	public int getMaxTestLen() {
		return maxTestLen;
	}

	public int getMaxRunLength() {
		return maxRunLength;
	}

	public int getNumTreeSplits() {
		return numTreeSplits;
	}

	public int getLeafSize() {
		return leafSize;
	}

	public int getNumTrees() {
		return numTrees;
	}

	public int getTreeIters() {
		return treeIters;
	}

	public int getMaxHistLen() {
		return maxHistLen;
	}

	public int getTestRuns() {
		return testRuns;
	}

	public double getEpsilon() {
		return epsilon;
	}

	public double getDiscount() {
		return discount;
	}

	public double getMinSingVal() {
		return minSingVal;
	}

	public String getPlanningType() {
		return planningType;
	}

	public String getModelLearnType() {
		return modelLearnType;
	}

	public Properties getPsrProperties() {
		return psrProperties;
	}

	public Properties getPlannerProperties() {
		return plannerProperties;
	}

	public DataSet getTestResults() {
		return testResults;
	}

	public List<List<List<ActionObservation>>> getTrainRunResults() {
		return trainRunResults;
	}

	public List<List<List<Double>>> getTrainRunRewards() {
		return trainRunRewards;
	}

	public List<Double> getModelBuildTimes() {
		return modelBuildTimes;
	}

	public List<Double> getPolicyConstructionTimes() {
		return policyConstructionTimes;
	}

	public List<List<Double>> getRandRewards() 
	{
		return randRewards;
	}


}
