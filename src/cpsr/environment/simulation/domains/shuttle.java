package cpsr.environment.simulation.domains;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import Parameter.Param;
import cpsr.environment.DataSet;
import cpsr.environment.components.Action;
import cpsr.environment.components.Observation;
import cpsr.environment.simulation.ASimulator;
import cpsr.model.POMDP;
import cpsr.planning.PSRPlanningExperiment;

public class shuttle extends ASimulator {
	protected static int NUM_ACTS = 3;
	protected static int NUM_OBS = 5;
	protected static int NUM_STS = 8;
	protected static int NUM_Rew = 3;
	public final static String[] Actions = {"TurnAround", "GoForward", "Backup"};
	public final static String[] Observations = {"See LRV forward", "See MRV forward", "See that we are docked in MRV", "See nothing", "See that we are docked in LRV"};
	public static String[] rewards = {"0.0", "10.0", "-3.0"};
	protected final String[] States = {"Docked in LRV", "Just outside space station MRV, front of ship facing station", "Space facing LRV", "Just outside space station LRV, back of ship facing station",
										"Just outside space station MRV, back of ship facing station", "Space, facing LRV", "Just outside space station LRV, front of ship facing station", "Docked in MRV"};
	protected final int[] IdxStates = {0, 1, 2, 3, 4, 5, 6, 7};
	protected final int[] IdxObservations = {0, 1, 2, 3, 4};
	protected double discount = 0.95;
	protected double[] belief_vector = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
	protected HashMap<Integer, double[][]> T_Matrix = new HashMap<Integer, double[][]>();
	protected HashMap<Integer, double[][]> O_Matrix = new HashMap<Integer, double[][]>();
	protected int Current_State = 0;
	//used to initialize Tiger State	
	public boolean inTerminalState;
	protected double currImmediateReward;
	protected int currObservation;
	protected int Actioncount;
	{
		//initial transition function
		double[][] T_TurnAround_Matrix = 
			  {{0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},  
			   {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0}, 
			   {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},  
			   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}, 
			   {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},  
			   {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},  
			   {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},  
			   {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
		double[][] T_GoForward_Matrix =
				  {{0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},  
				   {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},  
				   {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},  
				   {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 
				   {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},  
				   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0},  
				   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0},  
				   {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0}};
		double[][] T_Backup_Matrix = 
				  {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}, 
				   {0.0, 0.4, 0.3, 0.0, 0.3, 0.0, 0.0, 0.0}, 
				   {0.0, 0.0, 0.1, 0.8, 0.0, 0.0, 0.1, 0.0}, 
				   {0.7, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0}, 
				   {0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.7}, 
				   {0.0, 0.1, 0.0, 0.0, 0.8, 0.1, 0.0, 0.0}, 
				   {0.0, 0.0, 0.0, 0.3, 0.0, 0.3, 0.4, 0.0}, 
				   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
		this.T_Matrix.put(0, T_TurnAround_Matrix);
		this.T_Matrix.put(1, T_GoForward_Matrix);
		this.T_Matrix.put(2, T_Backup_Matrix);
		
		//initial observation function
		double[][] O_Matrix = 
			{{0.0, 0.0, 0.0, 0.0, 1.0}, 
			 {0.0, 1.0, 0.0, 0.0, 0.0}, 
			 {0.0, 0.7, 0.0, 0.3, 0.0}, 
			 {0.0, 0.0, 0.0, 1.0, 0.0}, 
			 {0.0, 0.0, 0.0, 1.0, 0.0}, 
			 {0.7, 0.0, 0.0, 0.3, 0.0}, 
			 {1.0, 0.0, 0.0, 0.0, 0.0}, 
			 {0.0, 0.0, 1.0, 0.0, 0.0}};
		this.O_Matrix.put(0, O_Matrix);
		this.O_Matrix.put(1, O_Matrix);
		this.O_Matrix.put(2, O_Matrix);
		
	}
	public POMDP generatePOMDP() throws Exception
	{
		String file = "shuttle.alpha";
		String file1 = "shuttle.pg";
		POMDP pomdp = new POMDP(file, file1, NUM_STS, NUM_OBS);
		pomdp.setbelief(belief_vector);
		pomdp.setTs(T_Matrix);
		Map<Integer, Map<Integer, double[][]>> Os = new HashMap<Integer, Map<Integer, double[][]>>();
		for (Integer actid: O_Matrix.keySet())
		{
			Map<Integer, double[][]> actOs = new HashMap<Integer, double[][]>();
			double[][] O = O_Matrix.get(actid);
			for (int i = 0; i < O[0].length; i++)
			{
				double[][] diagO = new double[NUM_STS][NUM_STS];
				for (int j = 0; j < O.length; j++)
				{
					diagO[j][j] = O[j][i];
				}
				actOs.put(i, diagO);
			}
			Os.put(actid, actOs);
		}
		pomdp.setOs(Os);
		return pomdp;
	}
	
	public static void main(String args[]) throws Exception
	{
		shuttle tiger = new shuttle(1000);
		Param.initialRandom(5);
		Param.read_Param();
		Param.actionSize = NUM_ACTS;
		Param.observationSize = NUM_OBS;
		Param.rewardSize = NUM_Rew;
		Param.GameName = tiger.getName();
		Param.introducedReward = false;
		Param.NormalizePredictedResult = false;
		Param.TruncatedLikelihoods = false;
		Param.RunningVersion = "V2";
		Param.FinalThreshold = 50;
		PSRPlanningExperiment experiment = new PSRPlanningExperiment("PSRConfigs/shuttle", "PlanningConfigs/shuttle", tiger);
		Param.POMDPAction = "Action";
		for (int i = 0; i < 12; i++)
		{
			experiment.EvalPOMDP();
			experiment.publishResults("Evaluate1\\Shuttle\\shuttleEvalResultGId0PI" + Integer.toString(i + 1));
		}
		
	}
	
	
//	public static void main(String args[]) throws Exception
//	{
//		System.out.print("Loading data");
//		Param.read_Param();
//		Param.actionSize = NUM_ACTS;
//		Param.observationSize = NUM_OBS;
//		Param.rewardSize = NUM_Rew;
//		shuttle tiger = new shuttle(1000);
//		Param.GameName = tiger.getName();
//		Param.introducedReward = false;
//		Param.NormalizePredictedResult = false;
//		Param.TruncatedLikelihoods = false;
//		System.out.print("Running Experiment");
//		String PSRPath = "Pureshuttle.json";
//		Param.isRandomInit = true;
//		System.out.println(args[0]);
//		System.out.println("The number of games training:" + args[1]);
//		Param.RunningVersion = args[2];
//		Param.StartingAtSpecialPosition = true;
//		if (Param.RunningVersion.equals("V1"))
//		{
//			System.out.println("the game ends when openning a door");
//		}
//		else if (Param.RunningVersion.equals("V2"))
//		{
//			System.out.println("the game ends after 50 actions");
//			Param.FinalThreshold = 50;
//		}
//		else if (Param.RunningVersion.equals("V3"))
//		{
//			System.out.println("the game ends after 30 actions with having open a door. Otherwise, ends at 50th actions");
//			Param.FinalThreshold = 50;
//			Param.WeakThreshold = 30;
//		}
//		for (int game = 0; game < Integer.parseInt(args[1]); game++)
//		{
//			Param.initialRandom(game);
//			PSRPlanningExperiment experiment = new PSRPlanningExperiment("PSRConfigs/shuttle", "PlanningConfigs/shuttle", tiger);
//			experiment.runExperiment(game, args[0], PSRPath);
//		}
//	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	public shuttle()
	{
		super();
	}
	
	public shuttle(int maxRunLength)
	{
		super(maxRunLength);
	}
	
	@Override
	public String getName()
	{
		return "shuttle";
	}
	
	@Override
	protected Observation getCurrentObservation(Random rando)
	{
		return Observation.GetObservation(this.currObservation, this.currImmediateReward);
	}

	@Override
	protected int[][] initRun(Random rando)
	{
		//Randomly set the tiger state
		this.Current_State = generate_random_number_given_probabilities(belief_vector, IdxStates, rando);
		this.inTerminalState = false;
		Actioncount=0;
		return null;
	}
	

	@Override
	protected int getNumberOfActions() 
	{
		return NUM_ACTS;
	}

	@Override
	protected int getNumberOfObservations() 
	{
		return NUM_OBS;
	}

	@Override
	protected boolean inTerminalState() 
	{
		return inTerminalState;
	}
	
	// randomly pick up an element from number array with a set of probabilities
	// p is the probabilities
	// number is an array of elements
	protected int generate_random_number_given_probabilities(double[] p, int[] state, Random rando)
	{
		if (state.length != p.length)
			System.out.println("The size of p and number should be equal in generate_random_number_given_probabilities");
		int size = state.length;
		int randint = rando.nextInt(100);
		double prob=0;
		for (int i=0; i<size; i++)
		{
			prob += p[i];
			if (randint < prob*100)
			{
				return state[i];
			}
		}
		return state[size-1];
	}
	
	@Override
	protected boolean executeAction(Action act, Random rando) 
	{
		int actid = act.getID();
		double[][] T_function = this.T_Matrix.get(actid);
		double[][] O_function = this.O_Matrix.get(actid);
		
		/////////////////////////////////////////////////////////////////////////////////////
		// Moves
		int Next_State = (int)this.generate_random_number_given_probabilities(T_function[Current_State], IdxStates, rando);
		
		/////////////////////////////////////////////////////////////////////////////////////
		// Reward
		currImmediateReward = 0.0;
		if (actid == 1 && Current_State == 1 && Next_State == 1)
		{
			currImmediateReward = -3.0;
		}
		else if (actid == 1 && Current_State == 6 && Next_State == 6)
		{
			currImmediateReward = -3.0;
		}
		else if (actid == 2 && Current_State == 3 && Next_State == 0) 
		{
			currImmediateReward = 10.0;
		}
		/////////////////////////////////////////////////////////////////////////////////////
		// Generate Observation
		double[] Observation_probabilities = O_function[Next_State];
		this.currObservation = (int) this.generate_random_number_given_probabilities(Observation_probabilities, IdxObservations, rando);
		Current_State = Next_State;
		/////////////////////////////////////////////////////////////////////////////////////
		// If the agent open a door, no matter what happens, the episode of this game ends

		Actioncount++; 
		if (Actioncount == Param.FinalThreshold)
		{
			Actioncount = 0;
			this.inTerminalState = true;
		}
		return true;
	}

	@Override
	protected double getCurrentReward() 
	{
		return this.currImmediateReward;
	}
}
