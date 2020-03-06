package cpsr.environment.simulation.domains;


import java.util.Random;
import Parameter.Param;

import java.util.HashMap;
import java.util.Map;

import cpsr.environment.components.Action;
import cpsr.environment.components.Observation;
import cpsr.environment.simulation.ASimulator;
import cpsr.model.POMDP;
import cpsr.planning.PSRPlanningExperiment;



public class Tiger95 extends ASimulator {
	protected static int NUM_ACTS = 3;
	protected static int NUM_OBS = 2;
	protected static int NUM_STS = 2;
	protected static int NUM_Rew = 3;
	public final static String[] Actions = {"Open-Left", "Open-Right", "Listen"};
	public final static String[] Observations = {"Tiger-Left", "Tiger-Right"};
	public static String[] rewards = {"-100.0", "10.0", "-1.0"};
	protected final String[] States = {"Tiger-Left", "Tiger-Right"};
	protected double discount = 0.95;
	protected double[] belief_vector = {0.5, 0.5};
	protected int[] idxState = {0, 1};
	protected HashMap<Integer, double[][]> T_Matrix = new HashMap<Integer, double[][]>();
	protected HashMap<Integer, double[][]> O_Matrix = new HashMap<Integer, double[][]>();
	protected HashMap<Integer, double[]> R_Matrix = new HashMap<Integer, double[]>();
	protected int Tiger_State = 0;
	//used to initialize Tiger State	
	public boolean inTerminalState;
	protected double currImmediateReward;
	protected int currObservation;
	protected int Actioncount;
	{
		//initial transition function
		double[][] T_Listen_Matrix = {{1.0, 0}, {0, 1.0}};
		double[][] T_Open_Left_Matrix = {{0.5, 0.5}, {0.5, 0.5}};
		double[][] T_Open_Right_Matrix = {{0.5, 0.5}, {0.5, 0.5}};
		this.T_Matrix.put(0, T_Open_Left_Matrix);
		this.T_Matrix.put(1, T_Open_Right_Matrix);
		this.T_Matrix.put(2, T_Listen_Matrix);
		
		//initial observation function
		double[][] O_Listen_Matrix = {{0.85, 0.15}, {0.15, 0.85}};
		double[][] O_Open_Left_Matrix = {{0.5, 0.5}, {0.5, 0.5}};
		double[][] O_Open_Right_Matrix = {{0.5, 0.5}, {0.5, 0.5}};
		this.O_Matrix.put(0, O_Open_Left_Matrix);
		this.O_Matrix.put(1, O_Open_Right_Matrix);
		this.O_Matrix.put(2, O_Listen_Matrix);
		
		//initial reward function
		double[] R_Listen_Matrix = {-1, -1};
		double[] R_Open_Left_Matrix = {-100, 10};
		double[] R_Open_Right_Matrix = {10, -100};
		this.R_Matrix.put(0, R_Open_Left_Matrix);
		this.R_Matrix.put(1, R_Open_Right_Matrix);
		this.R_Matrix.put(2, R_Listen_Matrix);
	}
//	public static void main(String args[]) throws Exception
//	{
//		Tiger95 tiger = new Tiger95(1000);
//		Param.initialRandom(3);
//		Param.read_Param();
//		Param.actionSize = NUM_ACTS;
//		Param.observationSize = NUM_OBS;
//		Param.rewardSize = NUM_Rew;
//		Param.GameName = tiger.getName();
//		Param.introducedReward = false;
//		Param.NormalizePredictedResult = false;
//		Param.TruncatedLikelihoods = false;
//		Param.RunningVersion = "V2";
//		Param.FinalThreshold = 50;
//		PSRPlanningExperiment experiment = new PSRPlanningExperiment("PSRConfigs/tiger95", "PlanningConfigs/tiger95", tiger);
//		Param.POMDPAction = "Action";
//		for (int i = 0; i < 16; i++)
//		{
//			experiment.EvalPOMDP();
//			experiment.publishResults("Evaluate1\\Tiger95\\Tiger95EvalResultGId0PI" + Integer.toString(i + 1));
//		}
//	}
	
	public static void main(String args[]) throws Exception
	{
		System.out.print("Loading data");
		Param.read_Param();
		Param.actionSize = NUM_ACTS;
		Param.observationSize = NUM_OBS;
		Param.rewardSize = NUM_Rew;
		Tiger95 tiger = new Tiger95(1000);
		Param.isRandomInit = true;
		Param.GameName = tiger.getName();
		
		Param.introducedReward = true;
		Param.planningType = "action";
		
		Param.NormalizePredictedResult = false;
		Param.TruncatedLikelihoods = false;
		System.out.print("Running Experiment");
		String PSRPath = null;
		if (Param.introducedReward)
		{
			PSRPath = "PSRTiger95WithRewards.json";
		}
		else
		{
			PSRPath = "PurePSR_Tiger95.json";
		}
		System.out.println(args[0]);
		System.out.println("The number of games training:" + args[1]);
		Param.RunningVersion = args[2];
		Param.StartingAtSpecialPosition = true;
		if (Param.RunningVersion.equals("V1"))
		{
			System.out.println("the game ends when openning a door");
		}
		else if (Param.RunningVersion.equals("V2"))
		{
			System.out.println("the game ends after 50 actions");
			Param.FinalThreshold = 50;
		}
		else if (Param.RunningVersion.equals("V3"))
		{
			System.out.println("the game ends after 30 actions with having open a door. Otherwise, ends at 50th actions");
			Param.FinalThreshold = 50;
			Param.WeakThreshold = 30;
		}
		for (int game = 0; game < Integer.parseInt(args[1]); game++)
		{
			Param.initialRandom(game);
			PSRPlanningExperiment experiment = new PSRPlanningExperiment("PSRConfigs/tiger95", "PlanningConfigs/tiger95", tiger);
			experiment.runExperiment(game, args[0], PSRPath);
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	public POMDP generatePOMDP() throws Exception
	{
		String file = "tiger.alpha";
		String file1 = "tiger.pg";
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
	public Tiger95()
	{
		super();
	}
	
	public Tiger95(int maxRunLength)
	{
		super(maxRunLength);
	}
	
	@Override
	public String getName()
	{
		return "Tiger95";
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
		this.Tiger_State = generate_random_number_given_probabilities(belief_vector, idxState, rando);
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
		double[] R_function = this.R_Matrix.get(actid);
		
		/////////////////////////////////////////////////////////////////////////////////////
		// Tiger moves
		int[] state = {0, 1};
		int new_tiger_state = (int)this.generate_random_number_given_probabilities(T_function[this.Tiger_State], state, rando);
		
		/////////////////////////////////////////////////////////////////////////////////////
		// Genreate instant reward
		this.currImmediateReward = R_function[this.Tiger_State];
		this.Tiger_State = new_tiger_state;
		/////////////////////////////////////////////////////////////////////////////////////
		// Generate Observation
		double[] Observation_probabilities = O_function[this.Tiger_State];
		int[] observation = {0, 1};
		this.currObservation = (int) this.generate_random_number_given_probabilities(Observation_probabilities, observation, rando);
		
		/////////////////////////////////////////////////////////////////////////////////////
		// If the agent open a door, no matter what happens, the episode of this game ends
		if (Param.RunningVersion.equals("V1"))
		{
			if (actid != 2)
			{
				this.inTerminalState = true;
			}
		}
		else if (Param.RunningVersion.equals("V2"))
		{
			Actioncount++; 
			if (Actioncount == Param.FinalThreshold)
			{
				Actioncount = 0;
				this.inTerminalState = true;
			}
		}
		if (actid != 2)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	@Override
	protected double getCurrentReward() 
	{
		return this.currImmediateReward;
	}
}
