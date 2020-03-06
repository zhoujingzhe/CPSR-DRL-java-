package cpsr.environment.simulation.domains;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import Parameter.Param;
import cpsr.environment.components.Action;
import cpsr.environment.components.Observation;
import cpsr.environment.simulation.ASimulator;
import cpsr.model.POMDP;
import cpsr.planning.PSRPlanningExperiment;

public class niceEnv extends ASimulator {
	protected static int NUM_ACTS = 3;	
	protected static int NUM_OBS = 5;
	protected static int NUM_STS = 5;
	protected static int NUM_REW = 5;
	public static final String[] Actions = {"Turn-Left", "Stay", "Turn-Right"};
	public static final String[] Observations = {"left-2-ob", "left-1-ob", "middle-ob", "right-1-ob", "right-2-ob"};
	protected final String[] States = {"left-2-st", "left-1-st", "middle-st", "right-1-st", "right-2-st"};
	public static String[] rewards = {"-50", "-10", "100", "-10", "-50"};
	protected final double[] beliefvector = {0, 0, 1, 0, 0};
	protected double discount = 0.99;
	protected HashMap<Integer, double[][]> T_Matrix = new HashMap<Integer, double[][]>();
	protected HashMap<Integer, double[][]> O_Matrix = new HashMap<Integer, double[][]>();
	protected HashMap<Integer, double[]> R_Matrix = new HashMap<Integer, double[]>();
	protected int Agent_State = 0;
	protected boolean inTerminalState;
	protected double currImmediateReward;
	protected int currObservation;
	private int Actioncount = 0;
//	public static void main(String args[]) throws Exception
//	{
//		niceEnv tiger = new niceEnv(1000);
//		Param.initialRandom(5);
//		Param.read_Param();
//		Param.actionSize = NUM_ACTS;
//		Param.observationSize = NUM_OBS;
//		Param.rewardSize = NUM_REW;
//		Param.GameName = tiger.getName();
//		Param.introducedReward = false;
//		Param.NormalizePredictedResult = false;
//		Param.TruncatedLikelihoods = false;
//		Param.RunningVersion = "V2";
//		Param.FinalThreshold = 50;
//		PSRPlanningExperiment experiment = new PSRPlanningExperiment("PSRConfigs/niceEnv", "PlanningConfigs/niceEnv", tiger);
//		for (int i = 0; i < 14; i++)
//		{
//			experiment.EvalPOMDP();
//			experiment.publishResults("Evaluate1\\niceEnv\\niceEnvEvalResultGId0PI" + Integer.toString(i + 1));
//		}
//	}
	public static void main(String args[]) throws Exception
	{
		System.out.print("Loading data");
		Param.read_Param();
		Param.actionSize = NUM_ACTS;
		Param.observationSize = NUM_OBS;
		Param.rewardSize = NUM_REW;
		niceEnv game = new niceEnv(1000);
		Param.isRandomInit=false;
		Param.GameName = game.getName();
		Param.introducedReward = false;
		Param.NormalizePredictedResult = false;
		Param.TruncatedLikelihoods = false;
		Param.RunningVersion="Null";
		System.out.print("Running Experiment");
		String PSRPath = "PurePSRNiceEnv.json";
		System.out.println(args[0]);
		System.out.println("The number of games training:" + args[1]);
		Param.FinalThreshold = Integer.parseInt(args[2]);
		Param.StartingAtSpecialPosition = true;
		Param.planningType = "action";
		for (int gidx = 0; gidx < Integer.parseInt(args[1]); gidx++)
		{
			Param.initialRandom(gidx);
			PSRPlanningExperiment experiment = new PSRPlanningExperiment("PSRConfigs/niceEnv", "PlanningConfigs/niceEnv", game);
			experiment.runExperiment(gidx, args[0], PSRPath);
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	{
		double[][] T_turn_left = { 
				{ 0.95, 0.05, 0, 0, 0}, 
				{ 0.7, 0.25, 0.05, 0, 0}, 
				{ 0, 0.7, 0.25, 0.05, 0},
				{ 0, 0, 0.7, 0.25, 0.05}, 
				{ 0, 0, 0, 0.7, 0.30}};
		double[][] T_stay = { 
				{ 0.8, 0.2, 0.0, 0.0, 0.0}, 
				{ 0.2, 0.6, 0.2, 0.0, 0.0},
				{ 0.0, 0.2, 0.6, 0.2, 0.0},
				{ 0.0, 0.0, 0.2, 0.6, 0.2},
				{ 0.0, 0.0, 0.0, 0.2, 0.8}};
		double[][] T_turn_right = { 
				{ 0.3, 0.7, 0.0, 0.0, 0.0}, 
				{ 0.05, 0.25, 0.7, 0.0, 0.0},
				{ 0.0, 0.05, 0.25, 0.7, 0.0},
				{ 0.0, 0.0, 0.05, 0.25, 0.7},
				{ 0.0, 0.0, 0.0, 0.05, 0.95 }};
		
//		double[][] T_turn_left = { 
//				{ 1.0, 0.0, 0, 0, 0}, 
//				{ 0.95, 0.05, 0, 0, 0}, 
//				{ 0, 0.95, 0.05, 0, 0},
//				{ 0, 0, 0.95, 0.05, 0}, 
//				{ 0, 0, 0, 0.95, 0.05}};
//		double[][] T_stay = { 
//				{ 0.95, 0.05, 0.0, 0.0, 0.0}, 
//				{ 0.05, 0.9, 0.05, 0.0, 0.0},
//				{ 0.0, 0.05, 0.9, 0.05, 0.0},
//				{ 0.0, 0.0, 0.05, 0.9, 0.05},
//				{ 0.0, 0.0, 0.0, 0.05, 0.95}};
//		
//		double[][] T_turn_right = { 
//				{ 0.05, 0.95, 0.0, 0.0, 0.0}, 
//				{ 0.0, 0.05, 0.95, 0.0, 0.0},
//				{ 0.0, 0.0, 0.05, 0.95, 0.0},
//				{ 0.0, 0.0, 0.0, 0.05, 0.95},
//				{ 0.0, 0.0, 0.0, 0.0, 1.0}};
		
		this.T_Matrix.put(0, T_turn_left);
		this.T_Matrix.put(1, T_stay);
		this.T_Matrix.put(2, T_turn_right);
		
		//initial observation function
		double[][] Observation = { 
				{ 0.8, 0.2, 0.0, 0.0, 0.0}, 
				{ 0.1, 0.8, 0.1, 0.0, 0.0},
				{ 0.0, 0.1, 0.8, 0.1, 0.0}, 
				{ 0.0, 0.0, 0.1, 0.8, 0.1},
				{ 0.0, 0.0, 0.0, 0.2, 0.8}};
		this.O_Matrix.put(0, Observation);
		this.O_Matrix.put(1, Observation);
		this.O_Matrix.put(2, Observation);
		//initial reward function
		double[] Rewards = {-50, -10, 100, -10, -50};
		this.R_Matrix.put(0, Rewards);
	}
	public POMDP generatePOMDP() throws Exception
	{
		String file = "niceEnv.alpha";
		String file1 = "niceEnv.pg";
		POMDP pomdp = new POMDP(file, file1, NUM_STS, NUM_OBS);
		pomdp.setbelief(beliefvector);
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
	public niceEnv(int maxRunLength) {
		super(maxRunLength);
	}
	
	@Override
	public String getName()
	{
		return "niceEnv";
	}
	
	@Override
	protected Observation getCurrentObservation(Random rando)
	{
		return Observation.GetObservation(this.currObservation, this.currImmediateReward);
	}

	@Override
	protected int[][] initRun(Random rando)
	{
		//Randomly starts the game
		int[] states = {0, 1, 2, 3, 4};
		this.Agent_State = generate_random_number_given_probabilities(beliefvector, states, rando);
		this.inTerminalState = false;
		Actioncount = 0;
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
		double[][] O_function = this.O_Matrix.get(0);
		double[] R_function = this.R_Matrix.get(0);
		
		/////////////////////////////////////////////////////////////////////////////////////
		// Agent moves
		int[] state = {0, 1, 2, 3, 4};
		int newAgent_State = this.generate_random_number_given_probabilities(T_function[Agent_State], state, rando);
		
		/////////////////////////////////////////////////////////////////////////////////////
		// Genreate instant reward
		this.currImmediateReward = R_function[newAgent_State];
		Agent_State = newAgent_State;
		
		/////////////////////////////////////////////////////////////////////////////////////

		int[] observation = {0, 1, 2, 3, 4};
		this.currObservation = this.generate_random_number_given_probabilities(O_function[Agent_State], observation, rando);
		
		Actioncount++;
		/////////////////////////////////////////////////////////////////////////////////////
		// If the agent has implemented 30 actions, ends the game
		if (Param.FinalThreshold < Actioncount)
		{
			inTerminalState = true;
		}
		return false;
	}

	@Override
	protected double getCurrentReward() 
	{
		return this.currImmediateReward;
	}
}
