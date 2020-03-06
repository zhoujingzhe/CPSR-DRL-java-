package cpsr.environment.simulation.domains;

import java.util.HashMap;
import java.util.Random;

import Parameter.Param;
import cpsr.environment.components.Action;
import cpsr.environment.components.Observation;
import cpsr.environment.simulation.ASimulator;
import cpsr.model.POMDP;
import cpsr.planning.PSRPlanningExperiment;

public class Maze extends ASimulator {
	protected static int NUM_ACTS = 5;	
	protected static int NUM_OBS = 6;
	protected static int NUM_STS = 11;
	protected static int NUM_REW = 3;
	public static final String[] Actions = {"Move-north", "Move-south", "Move-east", "Move-west", "Reset"};
	public static final String[] Observations = {"left", "right", "neither", "both", "good", "bad"};
	protected final int[] IdxObservations = {0, 1, 2, 3, 4, 5};
	protected final String[] States = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"};
	protected final int[] IdxStates = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	public static String[] rewards = {"-0.04", "10", "-100"};
	protected double discount = 0.95;
	protected HashMap<Integer, double[][]> T_Matrix = new HashMap<Integer, double[][]>();
	protected HashMap<Integer, double[][]> O_Matrix = new HashMap<Integer, double[][]>();
	protected HashMap<Integer, double[]> R_Matrix = new HashMap<Integer, double[]>();
	protected int Agent_State = 0;
	protected double[] beliefVector = {0.111111, 0.111111, 0.111111, 0.0, 0.111111, 0.111111, 0.0, 0.111112, 0.111111, 0.111111, 0.111111};
	protected boolean inTerminalState;
	protected double currImmediateReward;
	protected int currObservation;
	private int Actioncount = 0;
	
	public static void main(String args[]) throws Exception
	{
		System.out.print("Loading data");
		Param.read_Param();
		Param.actionSize = NUM_ACTS;
		Param.observationSize = NUM_OBS;
		Param.rewardSize = NUM_REW;
		Maze game = new Maze(1000);
		Param.isRandomInit = true;
		Param.GameName = game.getName();
		Param.planningType = "action";
		Param.introducedReward = false;
		Param.NormalizePredictedResult = false;
		Param.TruncatedLikelihoods = false;
		Param.RunningVersion="Null";
		System.out.print("Running Experiment");
		String PSRPath = "Maze1.json";
		System.out.println(args[0]);
		System.out.println("The number of games training:" + args[1]);
		Param.FinalThreshold = Integer.parseInt(args[2]);
		Param.StartingAtSpecialPosition = true;
		Param.MaximumQIteration = 15;
		for (int gidx = 0; gidx < Integer.parseInt(args[1]); gidx++)
		{
			Param.initialRandom(gidx);
			PSRPlanningExperiment experiment = new PSRPlanningExperiment("PSRConfigs/Maze", "PlanningConfigs/Maze", game);
			experiment.runExperiment(gidx, args[0], PSRPath);
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	{
		double[][] Move_North =
			{{0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			 {0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			 {0.0, 0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			 {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			 {0.8, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			 {0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0},
			 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
			 {0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0},
			 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1, 0.0},
			 {0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0, 0.1},
			 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.1}};
		double[][] Move_South = 
			{{0.1, 0.1, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			 {0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			 {0.0, 0.1, 0.0, 0.1, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0},
			 {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			 {0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0},
			 {0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.8, 0.0},
			 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
			 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0},
			 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1, 0.0},
			 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1},
			 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9}};
		double[][] Move_East = 
			{{0.1, 0.8, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			 {0.0, 0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			 {0.0, 0.0, 0.1, 0.8, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0},
			 {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			 {0.1, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0},
			 {0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0},
			 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
			 {0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.8, 0.0, 0.0},
			 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 0.0},
			 {0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.8},
			 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.9}};
		double[][] Move_West = 
			{{0.9, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			 {0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			 {0.0, 0.8, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0},
			 {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			 {0.1, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0},
			 {0.0, 0.0, 0.1, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.1, 0.0},
			 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
			 {0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0},
			 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0},
			 {0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.8, 0.1, 0.0},
			 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.8, 0.1}};
		double[][] l = 
			{{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			 {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			 {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			 {0.111111, 0.111111, 0.111111, 0.0, 0.111111, 0.111111, 0.0, 0.111112, 0.111111, 0.111111, 0.111111},
			 {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			 {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			 {0.111111, 0.111111, 0.111111, 0.0, 0.111111, 0.111111, 0.0, 0.111112, 0.111111, 0.111111, 0.111111},
			 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
			 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
			 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
			 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
		this.T_Matrix.put(0, Move_North);
		this.T_Matrix.put(1, Move_South);
		this.T_Matrix.put(2, Move_East);
		this.T_Matrix.put(3, Move_West);
		this.T_Matrix.put(4, l);
		
		//initial observation function
		double[][] Observation = { 
				{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
				{0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
				{0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
				{0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
				{0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
				{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
				{0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
				{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
				{0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
				{0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
				{0.0, 1.0, 0.0, 0.0, 0.0, 0.0}};
		double[][] Observation_l = {
				{0.25, 0.25, 0.25, 0.25, 0.0, 0.0},
				{0.25, 0.25, 0.25, 0.25, 0.0, 0.0},
				{0.25, 0.25, 0.25, 0.25, 0.0, 0.0},
				{0.25, 0.25, 0.25, 0.25, 0.0, 0.0},
				{0.25, 0.25, 0.25, 0.25, 0.0, 0.0},
				{0.25, 0.25, 0.25, 0.25, 0.0, 0.0},
				{0.25, 0.25, 0.25, 0.25, 0.0, 0.0},
				{0.25, 0.25, 0.25, 0.25, 0.0, 0.0},
				{0.25, 0.25, 0.25, 0.25, 0.0, 0.0},
				{0.25, 0.25, 0.25, 0.25, 0.0, 0.0},
				{0.25, 0.25, 0.25, 0.25, 0.0, 0.0}};
		this.O_Matrix.put(0, Observation);
		this.O_Matrix.put(1, Observation_l);
		
		//initial reward function
		double[] Rewards1 = {-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04};
		double[] Rewards2 = {-0.04, -0.04, -0.04, 10.0, -0.04, -0.04, -100.0, -0.04, -0.04, -0.04, -0.04};
		this.R_Matrix.put(0, Rewards1);
		this.R_Matrix.put(1, Rewards2);
	}
	
	public Maze(int maxRunLength) {
		super(maxRunLength);
	}
	
	@Override
	public String getName()
	{
		return "Maze";
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
		this.Agent_State = generate_random_number_given_probabilities(beliefVector, IdxStates, rando);
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
		double[][] O_function;
		double[] R_function;
		if (actid != 4)
		{
			O_function = this.O_Matrix.get(0);
			R_function = this.R_Matrix.get(0);
		}
		else
		{
			O_function = this.O_Matrix.get(1);
			R_function = this.R_Matrix.get(1);
		}
		
		/////////////////////////////////////////////////////////////////////////////////////
		// Agent moves
		
		int newAgent_State = generate_random_number_given_probabilities(T_function[Agent_State], IdxStates, rando);
		
		/////////////////////////////////////////////////////////////////////////////////////
		// Genreate instant reward
		this.currImmediateReward = R_function[Agent_State];
		Agent_State = newAgent_State;
		
		/////////////////////////////////////////////////////////////////////////////////////

		this.currObservation = generate_random_number_given_probabilities(O_function[Agent_State], IdxObservations, rando);
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

	@Override
	public POMDP generatePOMDP() throws Exception {
		// TODO Auto-generated method stub
		return null;
	}
}
