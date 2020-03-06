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

public class Stand_tiger extends ASimulator {
	/*
	 * Action: open-left open-middle open-right listen stand
	 */
	protected static int NUM_ACTS = 5;	
	/*
	 * Observation: tiger-left-stand tiger-middle-stand tiger-right-stand, tiger-left-sit tiger-middle-sit tiger-right-sit
	 */
	protected static int NUM_OBS = 6;
	/*
	 * State: tiger-left-stand tiger-middle-stand tiger-right-stand, tiger-left-sit tiger-middle-sit tiger-right-sit
	 */
	protected static int NUM_STS = 6;
	protected static int NUM_REW = 4;
	public static final String[] Actions = {"Open-Left", "Open-Middle", "Open-Right", "Listen", "Stand-Up"};
	public static final String[] Observations = {"tiger-left-sit", "tiger-middle-sit", "tiger-right-sit", "tiger-left-stand", "tiger-middle-stand", "tiger-right-stand"};
	protected final String[] States = {"tiger-left-sit", "tiger-middle-sit", "tiger-right-sit", "tiger-left-stand", "tiger-middle-stand", "tiger-right-stand"};
	public static String[] rewards = {"-1.0", "-100.0", "30.0", "-1000.0"};
	protected double discount = 0.95;
//	protected double[] belief = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	protected double[] belief = {0.333333333, 0.333333333, 0.333333333, 0.0, 0.0, 0.0};
//	protected double[] belief_vector;
	protected HashMap<Integer, double[][]> T_Matrix = new HashMap<Integer, double[][]>();
	protected HashMap<Integer, double[][]> O_Matrix = new HashMap<Integer, double[][]>();
	protected HashMap<Integer, double[]> R_Matrix = new HashMap<Integer, double[]>();
	protected int Tiger_State = 0;
	//used to initialize Tiger State
//	protected int[] state;
//	protected int[] observation;
	protected boolean inTerminalState;
	protected double currImmediateReward;
	protected int currObservation;
	private int Actioncount = 0;
	private boolean HavingOpen = false;
	
//	public static void main(String args[]) throws Exception
//	{
//		Param.read_Param();
//		Param.actionSize = NUM_ACTS;
//		Param.observationSize = NUM_OBS;
//		Param.rewardSize = NUM_REW;
//		Param.lengthOfGame = 50;
//		Stand_tiger tiger = new Stand_tiger(1000);
//		Param.GameName = tiger.getName();
//		Param.introducedReward = true;
//		Param.NormalizePredictedResult = false;
//		Param.TruncatedLikelihoods = true;
//		String Path = args[0];
//		File folder = new File(Path);
//		File[] listOfFiles = folder.listFiles();
//		Param.Eval = false;
//		for (int i = 0; i < listOfFiles.length; i++) {
//		  if (listOfFiles[i].isFile())
//		  {
//			Param.initialRandom(i);
//			PSRPlanningExperiment experiment = new PSRPlanningExperiment("PSRConfigs/standtiger", "PlanningConfigs/standtiger", tiger);
//		    System.out.println(listOfFiles[i].getName());
//			experiment.Eval(listOfFiles[i].getPath());
////			experiment.publishResults(Path + "/" + listOfFiles[i].getName().replace(".ser", ""));
//		  }
//		}
//	}
	
	public static void main(String args[]) throws Exception
	{
		System.out.print("Loading data");
		Param.read_Param();
		Param.actionSize = NUM_ACTS;
		Param.observationSize = NUM_OBS;
		Param.rewardSize = NUM_REW;
		Stand_tiger2 tiger = new Stand_tiger2(1000);
		Param.GameName = tiger.getName();
		Param.isRandomInit = true;
		Param.introducedReward = true;
		Param.NormalizePredictedResult = false;
		Param.TruncatedLikelihoods = false;
		System.out.print("Running Experiment");
		String PSRPath = "StandTigerWithRewards.json";
		System.out.println(args[0]);
		System.out.println("The number of games training:" + args[1]);
		Param.RunningVersion = args[2];
		Param.StartingAtSpecialPosition=true;
		Param.planningType = "action";
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
			PSRPlanningExperiment experiment = new PSRPlanningExperiment("PSRConfigs/standtiger", "PlanningConfigs/standtiger", tiger);
			experiment.runExperiment(game, args[0], PSRPath);
		}
	}
	public POMDP generatePOMDP() throws Exception
	{
		String file = null;
		String file1 = null;
		POMDP pomdp = new POMDP(file, file1, NUM_STS, NUM_OBS);
		pomdp.setbelief(belief);
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
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	{
		double[][] T_Listen_Matrix = { 
				{ 1.0, 0, 0, 0, 0, 0 }, 
				{ 0, 1.0, 0, 0, 0, 0 }, 
				{ 0, 0, 1.0, 0, 0, 0 },
				{ 0, 0, 0, 1.0, 0, 0 }, 
				{ 0, 0, 0, 0, 1.0, 0 },
				{ 0, 0, 0, 0, 0.0, 1.0 }};
		double[][] T_Open_Left_Matrix = { 
				{ 1.0, 0.0, 0.0, 0.0, 0, 0 }, 
				{ 0.0, 1.0, 0.0, 0.0, 0, 0 },
				{ 0.0, 0.0, 1.0, 0.0, 0, 0 },
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 },
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 },
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 }};
		
		double[][] T_Open_Middle_Matrix = { 
				{ 1.0, 0.0, 0.0, 0.0, 0, 0 }, 
				{ 0.0, 1.0, 0.0, 0.0, 0, 0 },
				{ 0.0, 0.0, 1.0, 0.0, 0, 0 },
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 },
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 },
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 }};
		
		double[][] T_Open_Right_Matrix = { 
				{ 1.0, 0.0, 0.0, 0.0, 0, 0 }, 
				{ 0.0, 1.0, 0.0, 0.0, 0, 0 },
				{ 0.0, 0.0, 1.0, 0.0, 0, 0 },
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 },
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 },
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 }};
		
		double[][] T_Stand_Up_Matrix = { 
				{ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 },
				{ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 }, 
				{ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 },
				{ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 },
				{ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 }, 
				{ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 }};

		this.T_Matrix.put(0, T_Open_Left_Matrix);
		this.T_Matrix.put(1, T_Open_Middle_Matrix);
		this.T_Matrix.put(2, T_Open_Right_Matrix);
		this.T_Matrix.put(3, T_Listen_Matrix);
		this.T_Matrix.put(4, T_Stand_Up_Matrix);
		
		//initial observation function
		double[][] O_Listen_Matrix = { 
				{ 0.75, 0.15, 0.1, 0.0, 0, 0 }, 
				{ 0.1, 0.8, 0.1, 0.0, 0, 0 },
				{ 0.1, 0.15, 0.75, 0, 0, 0 }, 
				{ 0, 0.0, 0.0, 0.33333333, 0.33333333, 0.33333333 },
				{ 0, 0.0, 0.0, 0.33333333, 0.33333333, 0.33333333 },
				{ 0, 0.0, 0.0, 0.33333333, 0.33333333, 0.33333333 } };
		double[][] O_Open_Left_Matrix = { 
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 },
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 },
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 },
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 },
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 },
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 } };
		double[][] O_Open_Middle_Matrix = { 
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 },
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 },
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 },
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 },
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 },
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 } };		
		double[][] O_Open_Right_Matrix = { 
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 },
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 },
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 },
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 },
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 },
				{ 0.33333333, 0.33333333, 0.33333333, 0, 0.0, 0.0 } };
		double[][] O_Stand_Up_Matrix = { 
				{ 0.0, 0.0, 0, 0.33333333, 0.33333333, 0.33333333 },
				{ 0.0, 0.0, 0, 0.33333333, 0.33333333, 0.33333333 },
				{ 0.0, 0.0, 0, 0.33333333, 0.33333333, 0.33333333 },
				{ 0.0, 0.0, 0, 0.33333333, 0.33333333, 0.33333333 },
				{ 0.0, 0.0, 0, 0.33333333, 0.33333333, 0.33333333 },
				{ 0.0, 0.0, 0, 0.33333333, 0.33333333, 0.33333333 } };
//		double[][] O_Stand_Up_Matrix = { 
//				{ 0.0, 0.0, 0, 0.5, 0.25, 0.25 },
//				{ 0.0, 0.0, 0, 0.25, 0.5, 0.25 },
//				{ 0.0, 0.0, 0, 0.25, 0.25, 0.5 },
//				{ 0.0, 0.0, 0, 0.5, 0.25, 0.25 },
//				{ 0.0, 0.0, 0, 0.25, 0.5, 0.25 },
//				{ 0.0, 0.0, 0, 0.25, 0.25, 0.5 } };
		
//		double[][] O_Stand_Up_Matrix = { 
//				{ 0.0, 0.0, 0, 0.33333333, 0.33333333, 0.33333333 },
//				{ 0.0, 0.0, 0, 0.33333333, 0.33333333, 0.33333333 },
//				{ 0.0, 0.0, 0, 0.33333333, 0.33333333, 0.33333333 },
//				{ 0.0, 0.0, 0, 0.5, 0.25, 0.25 },
//				{ 0.0, 0.0, 0, 0.25, 0.5, 0.25 },
//				{ 0.0, 0.0, 0, 0.25, 0.25, 0.5 } };
		
		this.O_Matrix.put(0, O_Open_Left_Matrix);
		this.O_Matrix.put(1, O_Open_Middle_Matrix);
		this.O_Matrix.put(2, O_Open_Right_Matrix);
		this.O_Matrix.put(3, O_Listen_Matrix);
		this.O_Matrix.put(4, O_Stand_Up_Matrix);
		
		//initial reward function
		double[] R_Listen_Matrix = {-1, -1, -1, -1, -1, -1};
		double[] R_Open_Left_Matrix = {-1000, -1000, -1000, -100, 30, 30};
		double[] R_Open_Middle_Matrix = {-1000, -1000, -1000, 30, -100, 30};
		double[] R_Open_Right_Matrix = {-1000, -1000, -1000, 30, 30, -100};
		double[] R_Stand_Up_Matrix = {-1, -1, -1, -1, -1, -1};
		this.R_Matrix.put(0, R_Open_Left_Matrix);
		this.R_Matrix.put(1, R_Open_Middle_Matrix);
		this.R_Matrix.put(2, R_Open_Right_Matrix);
		this.R_Matrix.put(3, R_Listen_Matrix);
		this.R_Matrix.put(4, R_Stand_Up_Matrix);

		// Generate Observation
	}
	
	public Stand_tiger(int maxRunLength)
	{
		super(maxRunLength);
	}
	
	@Override
	public String getName()
	{
		return "Stand_tiger";
	}
	
	@Override
	protected Observation getCurrentObservation(Random rando)
	{
		return Observation.GetObservation(this.currObservation, this.currImmediateReward);
	}

	@Override
	protected int[][] initRun(Random rando)
	{
		//initial belief state
//		this.belief_vector = initial_belief.clone();
		//Randomly set the tiger state
		int[] state = {0, 1, 2, 3, 4, 5};
		this.Tiger_State = generate_random_number_given_probabilities(belief, state, rando);
		this.inTerminalState = false;
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
		int[] state = {0, 1, 2, 3, 4, 5};
		int new_tiger_state = this.generate_random_number_given_probabilities(T_function[this.Tiger_State], state, rando);
		
		/////////////////////////////////////////////////////////////////////////////////////
		// Genreate instant reward
		this.currImmediateReward = R_function[this.Tiger_State];
		this.Tiger_State = new_tiger_state;
		
		/////////////////////////////////////////////////////////////////////////////////////

		int[] observation = {0, 1, 2, 3, 4, 5};
		this.currObservation = this.generate_random_number_given_probabilities(O_function[this.Tiger_State], observation, rando);
		
		/////////////////////////////////////////////////////////////////////////////////////
		// If the agent open a door, no matter what happens, the episode of this game ends
		if (Param.RunningVersion.equals("V1"))
		{
			if (actid == 0 || actid == 1 || actid == 2)
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
		if (actid == 3 || this.currImmediateReward == -1)
		{
			int h = 0;
			h = 1;
		}
		return true;
	}

	@Override
	protected double getCurrentReward() 
	{
		return this.currImmediateReward;
	}
}
