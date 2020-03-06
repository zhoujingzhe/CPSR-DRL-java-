
package cpsr.planning.ertapprox.actionensembles;
import java.util.Arrays;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import cpsr.environment.TrainingDataSet;
import cpsr.environment.components.Action;
import cpsr.environment.components.ActionObservation;
import cpsr.environment.simulation.domains.Maze;
import cpsr.environment.simulation.domains.Stand_tiger;
import cpsr.environment.simulation.domains.Tiger95;
import cpsr.environment.simulation.domains.niceEnv;
import cpsr.environment.simulation.domains.shuttle;
import cpsr.model.APSR;
import cpsr.model.components.PredictionVector;
import cpsr.planning.APSRPlanner;
import cpsr.planning.IQFunction;
import cpsr.planning.exceptions.PSRPlanningException;
import jparfor.Functor;
import jparfor.MultiThreader;
import jparfor.Range;
import Parameter.Param;
import afest.datastructures.tree.decision.erts.ERTTrainingPoint;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.jblas.DoubleMatrix;


public class ActionERTQPlannerDRaPS extends APSRPlanner {

	private static final long serialVersionUID = 8708755181159673282L;
	TrainingDataSet sampledData;
	double aDiscount;
	/*
	 * Using and initial probabilty distribution. 
	 */
	////////////////////////////////////////////////////////////////////////////////////////////////
	/*
	 * generate fixed value bound 
	 */
	static double[] Z_Bar_Values;
	{
		if (Z_Bar_Values == null)
		{
			int num_atoms = Param.num_atoms;
			double vmax = Param.vmax;
			double vmin = Param.vmin;
			double increment = (vmax-vmin)/Double.valueOf(num_atoms-1);
			Z_Bar_Values = new double[num_atoms];
			for (int index = 0; index < num_atoms; index++)
			{
				Z_Bar_Values[index] = vmin + increment * index;
			}
		}
	}
	/*
	 * initialize the uniform distribution or gaussian distribution
	 */
	static double[] initial_Prob_distribution;
	{
		if (initial_Prob_distribution == null)
		{
			String type = Param.initialized_type;
			int left_bound = Param.left_bound;
			int right_bound = Param.right_bound;
			int num_atoms = Param.num_atoms;
			initial_Prob_distribution = new double[num_atoms];
			if (type.equals("uniform"))
			{
				for (int index = left_bound; index < right_bound; index++)
				{
					initial_Prob_distribution[index] = 1/ Double.valueOf(right_bound - left_bound);
				}
			}
			else if (type.equals("gaussian"))
			{
				NormalDistribution dist = new NormalDistribution(0, 200);
				for (int index = left_bound; index < right_bound; index++)
				{
					double value = this.Z_Bar_Values[index];
					double Prob = dist.density(value);
					initial_Prob_distribution[index] = Prob;
				}
			}
			try {
				initial_Prob_distribution = Normalize_Distribution(initial_Prob_distribution);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	
	/*
	 * designed for quantile distribution
	 */
	static double[] Quantiles_Probs;
	{
		if (Quantiles_Probs == null)
		{
			int num_atoms = Param.num_atoms + 1;
			double vmax = 1.0;
			double vmin = 0.0;
			double[] Z = new double[num_atoms];
			double increment = (vmax-vmin)/Double.valueOf(num_atoms-1);
			for (int index = 0; index < num_atoms; index++)
			{
				Z[index] = vmin + increment * index;
			}
			this.Quantiles_Probs = new double[num_atoms-1];
			for (int i=0; i < num_atoms - 1; i++)
			{
				Quantiles_Probs[i] = (Z[i] + Z[i+1]) / 2;
			}
		}
	}
	/*
	 * designed for quantile distribution
	 */
	static double[] Init_Quantile_Distribution;
	{
		if (Init_Quantile_Distribution == null)
		{
			int num_atoms = Param.num_atoms;
			String type = Param.initialized_type;
			this.Init_Quantile_Distribution = new double[num_atoms];
			if (type.equals("uniform"))
			{
				double vmax = Param.vmax;
				double vmin = Param.vmin;
				double increment = (vmax-vmin)/Double.valueOf(num_atoms-1);
				for (int index = 0; index < num_atoms; index++)
				{
					this.Init_Quantile_Distribution[index] = vmin + increment * index;
				}
			}
			else if (type.equals("random"))
			{
				double max = Param.vmax;
				double min = Param.vmin;
				for (int i = 0; i < num_atoms; i++)
				{
					Init_Quantile_Distribution[i] = (Math.random() * ((max - min) + 1)) + min;
				}
				Arrays.sort(Init_Quantile_Distribution);
			}
		}
	}
	
	
	/*
	 * generate a fixed value array for C51
	 */
	private double[] Generate_value_Array()
	{
		return Z_Bar_Values;
	}
	
	/*
	 * generate a fixed prob array
	 */
	private double[] Generate_Quantile_Array()
	{
		return Quantiles_Probs;
	}
	
	/*
	 * generate an initial value array for Quantiles
	 */
	private double[] Init_Quantile_Distribution()
	{
		return Init_Quantile_Distribution;
	}
	
	/*
	 * generate an initial Prob array for C51
	 */
	private double[] Generate_initial_Prob_distribution()
	{
		return initial_Prob_distribution;
	}
	
	/////////////////////////////////////////////////////////////////////////////////////////////////
	/**
	 * Constructs fittedQ from PSR and DataSet.
	 * 
	 * @param psr The predictive state representation used in planning.
	 * @param data The data used to plan.
	 * @param type Type of ERT ensemble(s). Either "ActionEnsembles" or 
	 * "ActionEnsemble."
	 * In the ActionEnsemble case actions are treated as features, and in the 
	 * ActionEnsemble case separates ensembles are built for each action. 
	 * @deprecated
	 */
	public ActionERTQPlannerDRaPS(APSR psr, TrainingDataSet data) 
	{
		if(psr.getDataSet().getClass() != data.getClass())
		{
			throw new PSRPlanningException("DataSets used to learn reward" +
					" must be same type as one used to learn PSR");
		}
		this.psr = psr;
		this.data = data;
	}

	/**
	 * Constructs fittedQ from PSR and DataSet.
	 * 
	 * @param psr The predictive state representation used in planning.
	 * @param type Type of ERT ensemble(s). Either "ActionEnsembles" or 
	 * "ActionEnsemble."
	 * In the ActionEnsemble case actions are treated as features, and in the 
	 * ActionEnsemble case separates ensembles are built for each action. 
	 */
	public ActionERTQPlannerDRaPS(APSR psr) 
	{
		this.psr = psr;
		this.data = null;
	}
	public ActionERTQPlannerDRaPS(APSR psr, IQFunction qFunction1) 
	{
		this.psr = psr;
		this.data = null;
		this.qFunction = qFunction1;
	}

	/**
	 * Learns a policy using specified parameters.
	 * 
	 * @param data The data set used.
	 * @param runs Number of training runs to use when collecting initial data.
	 * @param iterations Number of iterations to use when training trees.
	 * @param treesPerEnsemble Number of trees per ensemble.
	 * @param k Number of splits to create at each inner node.  
	 * If k is null, then sqrt(number of attribute) will be used.
	 * @param nMin Specifies when trees stop growing if |set| < nMin at leaf, growing stops.
	 * @return ActionEnsembleQFunction learned using specified PSR, 
	 * DataSet,and tree ensemble parameters
	 * @throws Exception 
	 */
	public IQFunction learnQFunction(TrainingDataSet data, int runs, int iterations, int k, int nMin, int treesPerEnsemble, double pDiscount, int iter) throws Exception
	{
		if (!Param.introducedReward)
		{
			throw new Exception("Using DRaPS equation should introduced Reward as the part of reward!");
		}
		this.data = data;
		aDiscount = pDiscount;
		this.qFunction = learnQFunctionHelper(runs, iterations, treesPerEnsemble, k, nMin, iter);
		return qFunction;
	}
	/*
	 * sampling data to construct trees
	 */
	private TrainingDataSet SamplingData()
	{
//		double ratio = 1.0 / (data.getBatchNumber() + 1);
//		ratio = Math.max(ratio, 0.5);
		double ratio = 1.0;
		TrainingDataSet sampleddata = new TrainingDataSet(data, ratio);
		return sampleddata;
	}
	public ActionEnsemblesQFunction learnQFunctionHelper(int runs, int iterations, 
			int treesPerEnsemble, int k, int nMin, int iter) throws Exception
	{		
		sampledData = data;
		
		ActionEnsemblesFeatureBuilder featBuilder = new ActionEnsemblesFeatureBuilder(sampledData, psr);	
		// based on trajectories, generate the predictive state for each actions and observation
		PredictionVector.ClearPV_List();
		System.out.println("size of PredictionVectors:" + PredictionVector.getSizeOfPVList());
		ArrayList<PredictionVector> features = featBuilder.buildFeatures(runs);
		System.out.println("size of PredictionVectors:" + PredictionVector.getSizeOfPVList());
		/*
		 * Saving the features
		if (iter==1)
		{
//			CheckingFeatures(features);
//			FileOutputStream fileOut3 = new FileOutputStream("Feature"+".ser");
//			ObjectOutputStream out3 = new ObjectOutputStream(fileOut3);
//			out3.writeObject(features);
//			out3.close();
//			fileOut3.close();
		}
		 */
		ArrayList<Double> rewards = featBuilder.getRewards();
		ArrayList<Action> actions = featBuilder.getOrderedListOfActions();
		System.out.println("size of features:" + Integer.toString(features.size()));
		System.out.println("size of rewards:" + Integer.toString(rewards.size()));
		System.out.println("size of actions:" + Integer.toString(actions.size()));
		// Choosing a learning algorithm
		if (Param.C51)
		{
			System.out.println("Distritbuion!");
			// distributional form
			return new ActionEnsemblesQFunction(psr, learnERTEnsembleZFunctionForC51(features, rewards,
					actions, iterations, treesPerEnsemble, k, nMin, true, iter));
		}
		else
		{
			System.out.println("fitted-Q!");
			// fitted Q form
			return new ActionEnsemblesQFunction(psr, learnERTEnsembleQFunction(features, rewards,
					actions, iterations, treesPerEnsemble, k, nMin, iter));
		}
	}
	/*
	 * C51 algorithm
	 */
	private ArrayList<HashMap<Action, ActionERTEnsemble>> learnERTEnsembleZFunctionForC51(
			ArrayList<PredictionVector> features, ArrayList<Double> rewards,
			ArrayList<Action> actions, int iterations, int treesPerEnsemble, int k, int nMin, boolean Distribution, int iter) throws Exception
			{
				boolean isfirstiteration = true;
				int Pre_size = (int)(actions.size()/sampledData.getActionSet().size() * 1.15);
				ArrayList<Integer> testsLength = (ArrayList<Integer>) sampledData.getRunLengths();						
				ArrayList<List<Integer>> testsForThreads = new ArrayList<List<Integer>>();
				ArrayList<List<Action>> actionsForThreads = new ArrayList<List<Action>>();
				ArrayList<List<PredictionVector>> featuresForThreads = new ArrayList<List<PredictionVector>>();
				ArrayList<List<Double>> rewardsForThreads = new ArrayList<List<Double>>();
				HashMap<Action, ArrayList<PredictionVector>> featuresForActions = new HashMap<Action, ArrayList<PredictionVector>>();
				HashMap<Action, ArrayList<Double>> rewardsForActions = new HashMap<Action, ArrayList<Double>>();
				SplitFeaturesAndRewardsForEachAction(features, rewards, actions, featuresForActions, rewardsForActions);
				for (Action act: featuresForActions.keySet())
				{
					System.out.println(featuresForActions.get(act).size());
				}
				//Distributing the tasks for each of thread
				int num = Param.numThreadsForComputingTargets*(iter+1);
				while (testsLength.size() % num != 0)
				{
					num--;
				}
				int numThreadsForComputingTargets = Math.min(num, Param.MaximumThreadsForComputingQ);
				Assign_Jobs(features, rewards, actions, testsLength, numThreadsForComputingTargets, actionsForThreads, testsForThreads, featuresForThreads, rewardsForThreads);
				if (Distribution)
				{
					ArrayList<HashMap<Action, ActionERTEnsemble>> Array_actionEnsembles = null;
					long TimeStartingQIteration = 0;
					/*
					 * Q iterations is increasing when a new batch of data coming 
					 */
					int UIteration = Math.min(iterations * (iter+1), Param.MaximumQIteration);
					System.out.println("Taking" + Integer.toString(UIteration) + " Iterations!");
					for(int i = 0; i < UIteration; i++)
					{
						long StartingQIteration = System.currentTimeMillis();
						ArrayList<HashMap<Integer, ArrayList<HashMap<Action, ArrayList<Double>>>>> MultiTargets = null;
						if(isfirstiteration)
						{
							MultiTargets = MultiThreader.foreach(new Range(numThreadsForComputingTargets), new Functor<Integer, HashMap<Integer,ArrayList<HashMap<Action, ArrayList<Double>>>>>() {
								@Override
								public HashMap<Integer, ArrayList<HashMap<Action, ArrayList<Double>>>> function(Integer input) {
									// TODO Auto-generated method stub
									long TimeStartingEachThread = System.currentTimeMillis();
									List<Action> actionForThisThread = actionsForThreads.get(input);
									List<Integer> testForThisThread = testsForThreads.get(input);
									List<PredictionVector> featureForThisThread = featuresForThreads.get(input);
									List<Double> rewardForThisThread = rewardsForThreads.get(input);
									
									ArrayList<HashMap<Action, ArrayList<Double>>> target = null;
									try {
										// Computing Q value
										target = computeTargets(featureForThisThread, null, actionForThisThread, rewardForThisThread, testForThisThread, Pre_size);
									} catch (Exception e) {
										// TODO Auto-generated catch block
										e.printStackTrace();
										throw new RuntimeException(e);
									}
									HashMap<Integer, ArrayList<HashMap<Action, ArrayList<Double>>>> re = new HashMap<Integer, ArrayList<HashMap<Action,ArrayList<Double>>>>();
									re.put(input, target);
									long TimeEndingEachThread = System.currentTimeMillis();
									return re;
								}
							});	
							isfirstiteration = false;
						}
						else
						{
							ArrayList<HashMap<Action, ActionERTEnsemble>> Array_actionEnsembles_clone = Array_actionEnsembles;
							MultiTargets = MultiThreader.foreach(new Range(numThreadsForComputingTargets), new Functor<Integer, HashMap<Integer,ArrayList<HashMap<Action, ArrayList<Double>>>>>() {
								@Override
								public HashMap<Integer, ArrayList<HashMap<Action, ArrayList<Double>>>> function(Integer input) {
									// TODO Auto-generated method stub
									long TimeStartingEachThread = System.currentTimeMillis();
									List<Action> actionForThisThread = actionsForThreads.get(input);
									List<Integer> testForThisThread = testsForThreads.get(input);
									List<PredictionVector> featureForThisThread = featuresForThreads.get(input);
									List<Double> rewardForThisThread = rewardsForThreads.get(input);
									
									ArrayList<HashMap<Action, ArrayList<Double>>> target = null;
									try {
										target = computeTargets(featureForThisThread, Array_actionEnsembles_clone, actionForThisThread, rewardForThisThread, testForThisThread, Pre_size);
									} catch (Exception e) {
										// TODO Auto-generated catch block
										e.printStackTrace();
										throw new RuntimeException(e);
									}
									HashMap<Integer, ArrayList<HashMap<Action, ArrayList<Double>>>> re = new HashMap<Integer, ArrayList<HashMap<Action,ArrayList<Double>>>>();
									re.put(input, target);
									long TimeEndingEachThread = System.currentTimeMillis();
									//System.out.println("Time Spent on thread " + Integer.toString(input) + " is" + Double.toString((double) (TimeEndingEachThread - TimeStartingEachThread)/1000));
									return re;
								}
							});	
						}

						long TimeOnMiddleOFProcess = System.currentTimeMillis();
						System.out.println("The time on Computing Target:" + Double.toString((TimeOnMiddleOFProcess-StartingQIteration)/1000.0));
						// organize the multi-threads outputs 
						ArrayList<HashMap<Action, ArrayList<ArrayList<Double>>>> New_Z_Target = Convert_ThreadsOutputForC51(MultiTargets);

							// generating forests for each bar of distributions and each actions
						ArrayList<HashMap<Integer, HashMap<Action, ActionERTEnsemble>>> distributions = MultiThreader.foreach(new Range(New_Z_Target.size()), new Functor<Integer, HashMap<Integer, HashMap<Action, ActionERTEnsemble>>>() {
							@Override
							public HashMap<Integer, HashMap<Action, ActionERTEnsemble>> function(Integer input) {
								// New_Z_Target.get(index) returns the probability values for the bar-index
								HashMap<Action, ArrayList<ArrayList<Double>>> targets = New_Z_Target.get(input);
								HashMap<Integer, HashMap<Action, ActionERTEnsemble>> ret = new HashMap<Integer, HashMap<Action, ActionERTEnsemble>>();;
								try {
									HashMap<Action, ActionERTEnsemble> actionEnsembles = nthIteration(featuresForActions, targets, treesPerEnsemble, k, nMin);
									ret.put(input, actionEnsembles);
								} catch (Exception e) {
									e.printStackTrace();
									throw new RuntimeException(e);
								}
								return ret;
							}
						});
						ERTTrainingPoint.removeContentERTList();
						// organize the Forests
						Array_actionEnsembles = Convert(distributions);
						long EndIngQIteration = System.currentTimeMillis();
						System.out.println("The time on Constructing trees:" + Double.toString((EndIngQIteration-TimeOnMiddleOFProcess)/1000.0));
					}
					return Array_actionEnsembles;
				}
				return null;
			}
	/*
	 * Fitted Q algorithm
	 */
	private HashMap<Action, ActionERTEnsemble> learnERTEnsembleQFunction(ArrayList<PredictionVector> features, ArrayList<Double> rewards,
			ArrayList<Action> actions, int iterations, int treesPerEnsemble, int k, int nMin, int iter) throws ClassNotFoundException, IOException
			{
				boolean isfirstiteration = true;
				int Pre_size = (int) (actions.size()/sampledData.getActionSet().size() * 1.15);
				ArrayList<Integer> testsLength = (ArrayList<Integer>) sampledData.getRunLengths();						
				ArrayList<List<Integer>> testsForThreads = new ArrayList<List<Integer>>();
				ArrayList<List<Action>> actionsForThreads = new ArrayList<List<Action>>();
				ArrayList<List<PredictionVector>> featuresForThreads = new ArrayList<List<PredictionVector>>();
				ArrayList<List<Double>> rewardsForThreads = new ArrayList<List<Double>>();
				HashMap<Action, ArrayList<PredictionVector>> featuresForActions = new HashMap<Action, ArrayList<PredictionVector>>();
				HashMap<Action, ArrayList<Double>> rewardsForActions = new HashMap<Action, ArrayList<Double>>();
				SplitFeaturesAndRewardsForEachAction(features, rewards, actions, featuresForActions, rewardsForActions);
				//Distributing the tasks for each of thread
				int num = Param.numThreadsForComputingTargets*(iter+1);
				while (testsLength.size() % num != 0)
				{
					num--;
				}
				int numThreadsForComputingTargets = Math.min(num, Param.MaximumThreadsForComputingQ);
				
				Assign_Jobs(features, rewards, actions, testsLength, numThreadsForComputingTargets, actionsForThreads, testsForThreads, featuresForThreads, rewardsForThreads);
				HashMap<Action, ActionERTEnsemble> actionEnsembles = null;
				int UIteration = Math.min(iterations * (iter+1), Param.MaximumQIteration);
				System.out.println("Taking" + Integer.toString(UIteration) + " Iterations!");
				for(int i = 0; i < UIteration; i++)
				{
					long StartingQIteration = 0;
					StartingQIteration = System.currentTimeMillis();
					ArrayList<HashMap<Integer, HashMap<Action, ArrayList<Double>>>> MultiTargets = null;
					if(isfirstiteration)
					{
						MultiTargets = MultiThreader.foreach(new Range(numThreadsForComputingTargets), new Functor<Integer, HashMap<Integer,HashMap<Action, ArrayList<Double>>>>() {
							@Override
							public HashMap<Integer, HashMap<Action, ArrayList<Double>>> function(Integer input) {
								// TODO Auto-generated method stub
								long TimeStartingEachThread = System.currentTimeMillis();
								List<Action> actionForThisThread = actionsForThreads.get(input);
								List<Integer> testForThisThread = testsForThreads.get(input);
								List<PredictionVector> featureForThisThread = featuresForThreads.get(input);
								List<Double> rewardForThisThread = rewardsForThreads.get(input);
								
								HashMap<Action, ArrayList<Double>> target = null;
								try {
									target = computeTargets(featureForThisThread, null, actionForThisThread, rewardForThisThread, testForThisThread, "Fitted-Q", Pre_size);
								} catch (Exception e) {
									// TODO Auto-generated catch block
									e.printStackTrace();
									throw new RuntimeException(e);
								}
								HashMap<Integer, HashMap<Action, ArrayList<Double>>> re = new HashMap<Integer, HashMap<Action,ArrayList<Double>>>();
								re.put(input, target);
								long TimeEndingEachThread = System.currentTimeMillis();
								//System.out.println("Time Spent on thread " + Integer.toString(input) + " is" + Double.toString((double) (TimeEndingEachThread - TimeStartingEachThread)/1000));
								return re;
							}
						});	
						isfirstiteration = false;
					}
					else
					{
						HashMap<Action, ActionERTEnsemble> actionEnsembles_clone = actionEnsembles;
						MultiTargets = MultiThreader.foreach(new Range(numThreadsForComputingTargets), new Functor<Integer, HashMap<Integer,HashMap<Action, ArrayList<Double>>>>() {
							@Override
							public HashMap<Integer, HashMap<Action, ArrayList<Double>>> function(Integer input) {
								// TODO Auto-generated method stub
								long TimeStartingEachThread = System.currentTimeMillis();
								List<Action> actionForThisThread = actionsForThreads.get(input);
								List<Integer> testForThisThread = testsForThreads.get(input);
								List<PredictionVector> featureForThisThread = featuresForThreads.get(input);
								List<Double> rewardForThisThread = rewardsForThreads.get(input);
								
								HashMap<Action, ArrayList<Double>> target = null;
								try {
									target = computeTargets(featureForThisThread, actionEnsembles_clone, actionForThisThread, rewardForThisThread, testForThisThread, "Fitted-Q", Pre_size);
								} catch (Exception e) {
									// TODO Auto-generated catch block
									e.printStackTrace();
									throw new RuntimeException(e);
								}
								HashMap<Integer, HashMap<Action, ArrayList<Double>>> re = new HashMap<Integer, HashMap<Action,ArrayList<Double>>>();
								re.put(input, target);
								long TimeEndingEachThread = System.currentTimeMillis();
								//System.out.println("Time Spent on thread " + Integer.toString(input) + " is" + Double.toString((double) (TimeEndingEachThread - TimeStartingEachThread)/1000));
								return re;
							}
						});	
					}
					long TimeOnMiddleOFProcess = System.currentTimeMillis();
					System.out.println("The time on Computing Target:" + Double.toString((TimeOnMiddleOFProcess-StartingQIteration)/1000.0));
					HashMap<Action, ArrayList<ArrayList<Double>>> targets = Convert_ThreadsOutputForFittedQ(MultiTargets);
					actionEnsembles = nthIteration(featuresForActions, targets, treesPerEnsemble, k, nMin);
					ERTTrainingPoint.removeContentERTList();
					long EndIngQIteration = System.currentTimeMillis();
					System.out.println("The time on Constructing trees:" + Double.toString((EndIngQIteration-TimeOnMiddleOFProcess)/1000.0));
				}
				return actionEnsembles;
			}
	
	/*
	 * Projection operator on C51
	 * ZArray the predicted Array of value
	 * ZArray_Prime the updated Array of value
	 * Z_Prob the predicted Array of probabilities
	 * Increment the delta_z
	 */
	private double[] Projection_Between_Two_Distributions(double[] ZArray, double[] ZArray_Prime, double[] Z_Prob, double increment) throws Exception
	{
		int size = Z_Prob.length;
		if (size != ZArray.length || size != ZArray_Prime.length)
		{
			throw new Exception("The length of ZArray, ZArray_Prime, Z_Prob should be equal");
		}
		double[] Z_Prime_Prob = new double[size];
		int index_j=0;
		for (int i=0; i < ZArray.length; i++)
		{
			double value = ZArray[i];
			double Prob = Z_Prob[i];
			////////////////////////////////////////////////////////////
			// if the value is on the left side of Z_Array_Prime
			if (value <= ZArray_Prime[0])
			{
				Z_Prime_Prob[0] = Z_Prime_Prob[0] + Prob;
				continue;
			}
			////////////////////////////////////////////////////////////
			// if the value is on the right side of Z_Array_Prime
			if (value >= ZArray_Prime[size-1])
			{
				Z_Prime_Prob[size-1] = Z_Prime_Prob[size-1] + Prob;
				index_j++;
				continue;
			}
			boolean isAssign = false;
			////////////////////////////////////////////////////////////
			// if the value is in the middle of Z_Array_Prime
			for (; index_j< ZArray_Prime.length - 1; index_j++)
			{
				double value_prime = ZArray_Prime[index_j];
				if (value_prime <= value && value <= value_prime + increment)
				{
					double dist1 = value-value_prime;
					double dist2 = value_prime + increment - value;
					double frac_rate1 = dist2 / (dist1 + dist2);
					double frac_rate2 = dist1 / (dist1 + dist2);
					Z_Prime_Prob[index_j] = Z_Prime_Prob[index_j] + frac_rate1 * Prob;
					Z_Prime_Prob[index_j + 1] = Z_Prime_Prob[index_j + 1] + frac_rate2 * Prob;
					if (isAssign == true)
					{
						throw new Exception("One bar assigned twice!");
					}
					isAssign = true;
					break;
				}
			}
			if (isAssign == false)
			{
				throw new Exception("The bar doesn't assigned!");
			}
		}
		
		////////////////////////////////////////////////////////////////////////////////////////
		/// Normalize the Z_Prime_Prob
		double[] Normalized_Z_Prime_Prob = this.Normalize_Distribution(Z_Prime_Prob);
		return Normalized_Z_Prime_Prob;
	}
	/*
	 * calculating the score of a distribution
	 */
	public static double Score_of_Distribution(double[] dist) throws Exception
	{
		int num_atoms = dist.length;
		if (dist.length != num_atoms)
		{
			throw new Exception("The probability distribution should have equal length of a value distribution");
		}
		if (Param.C51)
		{
			double vmax = Param.vmax;
			double vmin = Param.vmin;
			double[] Z = new double[num_atoms];
			double increment = (vmax-vmin)/Double.valueOf(num_atoms-1);
			for (int index = 0; index < num_atoms; index++)
			{
				Z[index] = vmin + increment * index;
			}
			double sum = 0;
			for (int i = 0; i < num_atoms; i++)
			{
				sum = sum + Z[i] * dist[i];
			}
			sum = (double) Math.round(sum*100)/100;
			return sum;
		}
		else if (Param.Quantile_DRL)
		{
			double sum = 0.0;
			for (double value:dist)
			{
				sum = sum + value;
			}
			sum = (double)Math.round(sum/num_atoms * 100)/100;
			return sum;
		}
		else
		{
			throw new Exception("Something wrong on Score of Distribution!");
		}
	}
	
	/*
	 *  performing bellman equation : Z = r + \gamma * Z'
	 *  ZArray: predicted distribution
	 *  rewards: instantaneous reward
	 *  gamma: discount rate
	 */
	private double[] Distributional_Belleman_Update(double[] ZArray, double rewards, double gamma)
	{
		double[] ZArray_Prime = new double[ZArray.length];
		for(int i=0; i<ZArray.length; i++)
		{
			ZArray_Prime[i] = gamma*ZArray[i] + rewards;
		}
		return ZArray_Prime;
	}
	/*
	 * normalize the distribution
	 */
	private double[] Normalize_Distribution(double[] Z_Prime_Prob) throws Exception
	{
		double sum = 0;
		for(double d : Z_Prime_Prob)
		    sum += d;
		double[] Normalized_Z_Prime_Prob = Z_Prime_Prob;
		double sum1 = 0;
		for(int index=0; index < Z_Prime_Prob.length; index++)
		{
			Normalized_Z_Prime_Prob[index] =  Z_Prime_Prob[index] * (1/sum);
			sum1 += Normalized_Z_Prime_Prob[index];
		}
		sum1 = (double) Math.round(sum1 * 100) / 100;
		if (sum1 != 1)
		{
			throw new Exception("After Normalization, the sum is not 1");
		}
		return Normalized_Z_Prime_Prob;
	}
	/*
	 * organize features and rewards data for each actions
	 */
	private void SplitFeaturesAndRewardsForEachAction(ArrayList<PredictionVector> features, ArrayList<Double> rewards, ArrayList<Action> actions, HashMap<Action, ArrayList<PredictionVector>> featuresForActions, HashMap<Action, ArrayList<Double>> rewardsForActions)
	{
		if(features.size()!=actions.size() && features.size()!=rewards.size())
		{
			System.err.println("The size of features is not equal to actions!");
		}
		//initialize the featuresForActions
		for (Action act:sampledData.getActionSet())
		{
			featuresForActions.put(act, new ArrayList<PredictionVector>());
			rewardsForActions.put(act, new ArrayList<Double>());
		}
		for (int index = 0; index < actions.size(); index++)
		{
			featuresForActions.get(actions.get(index)).add(features.get(index));
			rewardsForActions.get(actions.get(index)).add(rewards.get(index));
		}
	}
	// organize the outputs for FittedQ
	private HashMap<Action, ArrayList<ArrayList<Double>>> Convert_ThreadsOutputForFittedQ(ArrayList<HashMap<Integer, HashMap<Action, ArrayList<Double>>>> MultiTargets)
	{
		HashMap<Action, ArrayList<ArrayList<Double>>> targets = new HashMap<Action, ArrayList<ArrayList<Double>>>();
		
		for (Action act: sampledData.getActionSet())
		{
			targets.put(act, new ArrayList<ArrayList<Double>>());
		}
		int index = 0;
		while(index < MultiTargets.size())
		{
			int count = 0;
			boolean isAssigned = false;
			for (HashMap<Integer, HashMap<Action, ArrayList<Double>>> SingleThreadOutput:MultiTargets)
			{
				if (SingleThreadOutput.containsKey(index))
				{
					if (count == 1)
					{
						System.err.println("Error on Convert_ThreadsOutputForFittedQ");
					}
					for (Action act : SingleThreadOutput.get(index).keySet())
					{
						if (!SingleThreadOutput.get(index).get(act).isEmpty())
							targets.get(act).add(SingleThreadOutput.get(index).get(act));
					}
					count++;
					isAssigned = true;
				}
			}
			if(isAssigned)
			{
				index++;
			}
		}
		return targets;
	}
	// organize the outputs for C51
	private ArrayList<HashMap<Action, ArrayList<ArrayList<Double>>>> Convert_ThreadsOutputForC51(ArrayList<HashMap<Integer, ArrayList<HashMap<Action, ArrayList<Double>>>>> MultiTargets)
	{
		ArrayList<HashMap<Action, ArrayList<ArrayList<Double>>>> targets = new ArrayList<HashMap<Action, ArrayList<ArrayList<Double>>>>();
		for (int index = 0; index < Param.num_atoms; index++)
		{
			targets.add(new HashMap<Action, ArrayList<ArrayList<Double>>>());
			for (Action act: sampledData.getActionSet())
			{
				targets.get(index).put(act, new ArrayList<ArrayList<Double>>());
			}
		}
		int index = 0;
		while(index < MultiTargets.size())
		{
			int count = 0;
			boolean isAssigned = false;
			for (HashMap<Integer, ArrayList<HashMap<Action, ArrayList<Double>>>> SingleThreadOutput:MultiTargets)
			{
				if (SingleThreadOutput.containsKey(index))
				{
					if (count == 1)
					{
						System.err.println("Error on Convert_ThreadsOutputForC51");
					}
					for (int idx = 0; idx < SingleThreadOutput.get(index).size(); idx++)
					{
						for (Action act : SingleThreadOutput.get(index).get(idx).keySet())
						{
							if (!SingleThreadOutput.get(index).get(idx).get(act).isEmpty())
								targets.get(idx).get(act).add(SingleThreadOutput.get(index).get(idx).get(act));
						}
					}
					count++;
					isAssigned = true;
				}
			}
			if (isAssigned)
			{
				index++;
			}
		}
		return targets;
	}
	// distribute tasks for each threads
	private void Assign_Jobs(ArrayList<PredictionVector> features, ArrayList<Double> rewards, ArrayList<Action> actions, ArrayList<Integer> testsLength, int num_Threads, ArrayList<List<Action>> actionsForThreads, ArrayList<List<Integer>> testsForThreads, ArrayList<List<PredictionVector>> featuresForThreads, ArrayList<List<Double>> rewardsForThreads)
	{
		if(features.size() != rewards.size() || actions.size() != features.size())
		{
			System.err.println("Error! on Assigned Jobs");
		}
		int WorksForSingleThread = testsLength.size() / num_Threads;
		for (int index = 0; index<num_Threads; index++)
		{
			if (index == num_Threads - 1)
			{
				List<Integer> testsForSingleThread = testsLength.subList(index*WorksForSingleThread, testsLength.size());
				
				testsForThreads.add(testsForSingleThread);
				int sum = 0;
				for (Integer s:testsForSingleThread)
				{
					sum+=s;
				}
				int fromIndex = 0;
				for (List<Action> singleThreadActions:actionsForThreads)
				{
					fromIndex += singleThreadActions.size();
				}
				actionsForThreads.add(actions.subList(fromIndex, fromIndex + sum));
				rewardsForThreads.add(rewards.subList(fromIndex, fromIndex + sum));
				featuresForThreads.add(features.subList(fromIndex, fromIndex + sum));
			}
			else {
				List<Integer> testsForSingleThread = testsLength.subList(index*WorksForSingleThread, (index+1)*WorksForSingleThread);
				testsForThreads.add(testsForSingleThread);
				int sum = 0;
				for (Integer s:testsForSingleThread)
				{
					sum+=s;
				}
				int fromIndex = 0;
				for (List<Action> singleThreadActions:actionsForThreads)
				{
					fromIndex += singleThreadActions.size();
				}
				try {
					actionsForThreads.add(actions.subList(fromIndex, fromIndex + sum));
					rewardsForThreads.add(rewards.subList(fromIndex, fromIndex + sum));
					featuresForThreads.add(features.subList(fromIndex, fromIndex + sum));
				}catch (IndexOutOfBoundsException e)
				{
					System.err.println("Error on AssignJobs!");
				}
				
			}
		}
	}
	
	/**
	 * Helper method performs one iteration of learning.
	 *
	 * @param features Features used to train regression ensemble.
	 * @param targets Targets (labels) used to train regression ensemble. 
	 * @param iterations Number of iterations to use when training trees.
	 * @param treesPerEnsemble Number of trees per ensemble.
	 * @param k Number of splits to create at each inner node.  If k is null, 
	 * then sqrt(number of attribute) will be used.
	 * @param nMin Specifies when trees stop growing if |set| < nMin at leaf, growing stops.
	 * @return Mapping of actions to ActionERTEnsembles
	 */
	private HashMap<Action, ActionERTEnsemble> nthIteration(HashMap<Action, ArrayList<PredictionVector>> features,HashMap<Action, ArrayList<ArrayList<Double>>> targets, 
			int treesPerEnsemble, int k, int nMin)
			{
				HashMap<Action, ActionERTEnsemble> actionEnsembles = new HashMap<Action, ActionERTEnsemble>();
				Object[] actions = sampledData.getActionSet().toArray();
				MultiThreader.foreach(new Range(actions.length), new Functor<Integer, Object>() {
					@Override
					public Object function(Integer input) {
						Action act = (Action) actions[input];
						ActionERTEnsembleGrower grower = new ActionERTEnsembleGrower();
						try {
							grower.setTrainData(features.get(act), targets.get(act));
						} catch(Exception e)
						{
							e.printStackTrace();
						}
						ActionERTEnsemble ret = null;
						try {
							ret = grower.growActionERTEnsemble(k, treesPerEnsemble, nMin, act);
						} catch (Exception e)
						{
							e.printStackTrace();
						} 
						synchronized (actionEnsembles) {
							actionEnsembles.put(act, ret);
						}
						return null;
					}
				});
				return actionEnsembles;
			}
	private void UpdateActionERTEnsemble(HashMap<Action, ArrayList<PredictionVector>> features,HashMap<Action, ArrayList<ArrayList<Double>>> targets, HashMap<Action, ActionERTEnsemble> actionEnsembles)
	{
		Object[] actions = sampledData.getActionSet().toArray();
		MultiThreader.foreach(new Range(actions.length), new Functor<Integer, Object>() {
			@Override
			public Object function(Integer input) {
				Action act = (Action) actions[input];
				ArrayList<PredictionVector> pvs = features.get(act);
				ArrayList<ArrayList<Double>> values = targets.get(act);
				ActionERTEnsemble actionEnsemble = actionEnsembles.get(act);
				int secondIndex = 0;
				int firstIndex = 0;
				for (int index = 0; index < pvs.size(); index++)
				{
					PredictionVector pv = pvs.get(index);
					double value = values.get(firstIndex).get(secondIndex);
					secondIndex = (secondIndex+1)%values.get(firstIndex).size();
					if (secondIndex == 0)
					{
						firstIndex++;
					}
					actionEnsemble.updateTreesByPV(value, pv);
				}
				return null;
			}
		});
	}

	/**
	 * Computes targets used in Q-iteration.
	 * 
	 * @param features Features of each point.
	 * @param actionEnsembles Mapping of actions to ActionERTEnsembles.
	 * @param actions Ordered list of actions.
	 * @param rewards Ordered list of immediate rewards. 
	 * @return Mapping of actions to discounted rewards (i.e. targets).
	 * @throws Exception 
	 */
	// Q-Learning
	private HashMap<Action, ArrayList<Double>> computeTargets(List<PredictionVector> features, 
			HashMap<Action, ActionERTEnsemble> actionEnsembles, List<Action> actions,
			List<Double> rewards, List<Integer> testsLength, String name, int Pre_size) throws Exception
			{
				HashMap<Action, ArrayList<Double>> targets =  new HashMap<Action,ArrayList<Double>>();
				intializeMapsForComputeTargets(targets, Pre_size);
				int inRunCount = 1;
				int runCount = 0;
				for(int i = 0; i < actions.size(); i++)
				{
					double maxExpectedValue = -Double.MAX_VALUE;
					Action act1 = actions.get(i);
					if(actionEnsembles!=null && !(inRunCount == testsLength.get(runCount)))
					{
						PredictionVector CurrentState = features.get(i);
						Map<ActionObservation, Double> AllPreds= psr.getAllPrediction(CurrentState, act1);
						for (ActionObservation actob: AllPreds.keySet())
						{
							if (act1.getID() != actob.getAction().getID())
							{
								throw new Exception("ActionID are different!");
							}
							PredictionVector NextState = psr.get_pv(actob);
							double r;
							if (Param.GameName.equals("Tiger95"))
							{
								r = Double.parseDouble(Tiger95.rewards[actob.getObservation().getrID()]);
							}
							else if (Param.GameName.equals("StandTiger"))
							{
								r = Double.parseDouble(Stand_tiger.rewards[actob.getObservation().getrID()]);
							}
							else if (Param.GameName.equals("Maze"))
							{
								r = Double.parseDouble(Maze.rewards[actob.getObservation().getrID()]);
							}
							else if (Param.GameName.equals("shuttle"))
							{
								r = Double.parseDouble(shuttle.rewards[actob.getObservation().getrID()]);
							}
							else if (Param.GameName.equals("niceEnv"))
							{
								r = Double.parseDouble(niceEnv.rewards[actob.getObservation().getrID()]);
							}
							else
							{
								throw new Exception("There is no such game!");
							}
							double prob = AllPreds.get(actob);
							double maxNextValue = -Double.MAX_VALUE;
							for(Action act : sampledData.getActionSet())
							{
								double value = actionEnsembles.get(act).getValueEstimate(NextState);
								maxNextValue = Math.max(value, maxNextValue);
							}
							maxExpectedValue += (r + maxNextValue) * prob;
						}
					}
					
					if(inRunCount == testsLength.get(runCount))
					{
						runCount++;
						inRunCount = 0;
					}
					inRunCount++;
					double qvalue = rewards.get(i);
					if (actionEnsembles!=null)
					{
						qvalue = (double)(Math.round(maxExpectedValue * 1000000.0)) / 1000000.0;
					}
					targets.get(act1).add(qvalue);
				}
				return targets;
			}
	// C51
	private ArrayList<HashMap<Action, ArrayList<Double>>> computeTargets(List<PredictionVector> features, 
			ArrayList<HashMap<Action, ActionERTEnsemble>> Array_actionEnsembles, List<Action> actions,
			List<Double> rewards, List<Integer> testsLength, int Pre_size) throws Exception
			{
				ArrayList<HashMap<Action, ArrayList<Double>>> targets = new ArrayList<HashMap<Action, ArrayList<Double>>>();
				intializeMapsForComputeTargets(targets, Pre_size);
				int inRunCount = 1;
				int runCount = 0;
				int num_atoms = Param.num_atoms;
				double vmin = Param.vmin;
				double vmax = Param.vmax;
				double increment = (vmax - vmin) / Double.valueOf(num_atoms - 1);

				double[] Z = Generate_value_Array();
				for(int i = 0; i < actions.size(); i++)
				{
					Action act1 = actions.get(i);			
					double[] maxProbDist = null;
					if(Array_actionEnsembles!=null&&!(inRunCount == testsLength.get(runCount)))
					{
						maxProbDist = new double[Param.num_atoms];
						PredictionVector CurrentState = features.get(i);
						Map<ActionObservation, Double> AllPreds = psr.getAllPrediction(CurrentState, act1);
						for (ActionObservation actob: AllPreds.keySet())
						{
							if (act1.getID() != actob.getAction().getID())
							{
								throw new Exception("ActionIDs are different!");
							}
							PredictionVector NextState = psr.get_pv(actob);
							double prob = AllPreds.get(actob);
							double r;
							if (Param.GameName.equals("Tiger95"))
							{
								r = Double.parseDouble(Tiger95.rewards[actob.getObservation().getrID()]);
							}
							else if (Param.GameName.equals("StandTiger"))
							{
								r = Double.parseDouble(Stand_tiger.rewards[actob.getObservation().getrID()]);
							}
							else if (Param.GameName.equals("Maze"))
							{
								r = Double.parseDouble(Maze.rewards[actob.getObservation().getrID()]);
							}
							else if (Param.GameName.equals("shuttle"))
							{
								r = Double.parseDouble(shuttle.rewards[actob.getObservation().getrID()]);
							}
							else if (Param.GameName.equals("niceEnv"))
							{
								r = Double.parseDouble(niceEnv.rewards[actob.getObservation().getrID()]);
							}
							else
							{
								throw new Exception("There is no such game!");
							}

							double[] maxNextState_Prob = null;
							double maxscore = -Double.MAX_VALUE;
							for(Action act : sampledData.getActionSet())
							{
								double[] dist1 = new double[Array_actionEnsembles.size()];
								for (int idx = 0; idx < Array_actionEnsembles.size(); idx++)
								{
									HashMap<Action, ActionERTEnsemble> actionEnsembles = Array_actionEnsembles.get(idx);
									dist1[idx] = actionEnsembles.get(act).getValueEstimate(NextState);
								}
								double score = Score_of_Distribution(dist1);
								if (score > maxscore)
								{
									maxscore = score;
									maxNextState_Prob = dist1;
								}
							}
							double[] Z_Prime = Distributional_Belleman_Update(Z, r, aDiscount);
							double[] dist2 = Projection_Between_Two_Distributions(Z_Prime, Z, maxNextState_Prob, increment);
							for (int j = 0; j < dist2.length; j++)
							{
								maxProbDist[j] += (dist2[j] * prob);
							}
						}
						
					}
					if(inRunCount == testsLength.get(runCount))
					{
						runCount++;
						inRunCount = 0;
					}
					inRunCount++;
					//////////////////////////////////////////////////////////////////////////////////////////////////////////
					if (maxProbDist==null)
					{
						double[] Z_Prob = Generate_initial_Prob_distribution();
						double[] Z_Prime = Distributional_Belleman_Update(Z, rewards.get(i), aDiscount);
						maxProbDist = Projection_Between_Two_Distributions(Z_Prime, Z, Z_Prob, increment);
					}
					for (int idx = 0; idx < maxProbDist.length; idx++)
					{
						targets.get(idx).get(act1).add(maxProbDist[idx]);
					}
				}
				return targets;
			}

	/**
	 * Intializes maps for method computeTargets()
	 * 
	 * @param targets The targets map.
	 * @param actionCounters Action counter map.
	 * @param actions Ordered list of actions.
	 */
	private void intializeMapsForComputeTargets(HashMap<Action, ArrayList<Double>> targets, int Pre_size)
	{
		for(Action act : sampledData.getActionSet())
		{
			targets.put(act, new ArrayList<Double>(Pre_size));
		}
	}
	private void intializeMapsForComputeTargets(ArrayList<HashMap<Action, ArrayList<Double>>> targets, int Pre_size)
	{
		int num_atoms = Param.num_atoms;
		for (int i = 0; i < num_atoms; i++)
		{
			HashMap<Action, ArrayList<Double>> A = new HashMap<Action, ArrayList<Double>>();
			for(Action act : sampledData.getActionSet())
			{
				A.put(act, new ArrayList<Double>(Pre_size));
			}
			targets.add(A);
		}
	}
	/*
	 * organize  the multithreads outputs of the decision trees
	 */
	private ArrayList<HashMap<Action, ActionERTEnsemble>> Convert(ArrayList<HashMap<Integer, HashMap<Action, ActionERTEnsemble>>> distribution) throws Exception
	{
		ArrayList<HashMap<Action, ActionERTEnsemble>> Array_ActionEnsembles = new ArrayList<HashMap<Action, ActionERTEnsemble>>();
		for (int index = 0; index < Param.num_atoms; index++)
		{
			Array_ActionEnsembles.add(new HashMap<Action, ActionERTEnsemble>());
		}
		for (HashMap<Integer, HashMap<Action, ActionERTEnsemble>> bar : distribution)
		{
			int count = 0;
			for (Integer index : bar.keySet())
			{
				if(count==1)
				{
					throw new Exception("Error on Convert!");
				}
				Array_ActionEnsembles.set(index, bar.get(index));
				count ++;
			}
		}
		return Array_ActionEnsembles;
	}
	/*
	 * debug function for checking if the policy are same
	 */
	void CheckingActionEnsembles(HashMap<Action, ActionERTEnsemble> actionEnsembles, int i) throws IOException, ClassNotFoundException
	{
		FileInputStream fileIn = new FileInputStream("ActionEnsemble"+Integer.toString(i)+".ser");
        ObjectInputStream in = new ObjectInputStream(fileIn);
        HashMap<Action, ActionERTEnsemble> actionEnsembles1 = (HashMap<Action, ActionERTEnsemble>) in.readObject();
        in.close();
        fileIn.close();
		FileInputStream fileIn1 = new FileInputStream("Feature.ser");
        ObjectInputStream in1 = new ObjectInputStream(fileIn1);
        ArrayList<PredictionVector> pv_list = (ArrayList<PredictionVector>) in1.readObject();
        in1.close();
        fileIn1.close();
        for (Action key: actionEnsembles.keySet())
        {
        	for (PredictionVector pv : pv_list.subList(300, 800))
        	{
        		double value1 = actionEnsembles.get(key).getValueEstimate(pv);
        		double value2 = actionEnsembles1.get(key).getValueEstimate(pv);
        		value1 = ((double)(Math.round(value1)*10000))/10000;
        		value2 = ((double)(Math.round(value2)*10000))/10000;
        		if (value1!=value2)
        		{
        			System.out.println("Different");
        		}
        	}
        }
	}
	/*
	 * debug functions for checking Qvalue for each ao
	 */
	void CheckingQvalueOfActionObservation(HashMap<Action, ArrayList<ArrayList<Double>>> MultiTarget, int i) throws IOException, ClassNotFoundException
	{
		FileInputStream fileIn = new FileInputStream("QValues"+Integer.toString(i)+".ser");
        ObjectInputStream in = new ObjectInputStream(fileIn);
        HashMap<Action, ArrayList<ArrayList<Double>>> MultiTarget1 = (HashMap<Action, ArrayList<ArrayList<Double>>>) in.readObject();
        in.close();
        fileIn.close();
    	for (Action key: MultiTarget.keySet())
    	{
    		ArrayList<ArrayList<Double>> ActionValue1 = MultiTarget.get(key);
    		ArrayList<ArrayList<Double>> ActionValue2 = MultiTarget1.get(key);
    		for (int idx = 0; idx < ActionValue1.size(); idx++)
    		{
    			ArrayList<Double> values1 = ActionValue1.get(idx);
    			ArrayList<Double> values2 = ActionValue2.get(idx);
    			if (values1.size()!=values2.size())
    			{
    				System.out.println("Different!");
    			}
    			for (int idx1= 0; idx1 < values1.size(); idx1++)
    			{
    				double v1 = ((double)(Math.round(values1.get(idx1)*100)))/100;
    				double v2 = ((double)(Math.round(values2.get(idx1)*100)))/100;
    				if (v1 != v2)
    				{
    					System.out.println("Different!");
    				}
    			}
    		}
    	}
	}
	
	/*
	 * debug function for checking if the features are different.
	 */
	void CheckingFeatures(ArrayList<PredictionVector> aoMat) throws IOException, ClassNotFoundException
	{
		FileInputStream fileIn = new FileInputStream("Feature.ser");
        ObjectInputStream in = new ObjectInputStream(fileIn);
        ArrayList<PredictionVector> aoMat1 = (ArrayList<PredictionVector>) in.readObject();
        in.close();
        fileIn.close();
        
       
        if (aoMat.size()!= aoMat1.size())
        {
        	System.out.println("Different1");
        }
        for (int i = 0; i < aoMat.size(); i++)
        {
            DoubleMatrix Mat1 = aoMat.get(i).getVector();
            DoubleMatrix Mat2 = aoMat1.get(i).getVector();
            if (!Mat1.equals(Mat2))
            {
            	System.out.println("Different");
            }
        }
	}

}
