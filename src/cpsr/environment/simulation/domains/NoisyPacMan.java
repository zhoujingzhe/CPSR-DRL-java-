package cpsr.environment.simulation.domains;

import java.util.Random;

import Parameter.Param;
import cpsr.environment.components.Observation;
import cpsr.planning.PSRPlanningExperiment;

public class NoisyPacMan extends PacMan 
{
	
	public static double NOISE = 0.5;
	
	private static int NUM_OBS = (int)1<<12;

	public static void main(String args[]) throws Exception
	{
		NoisyPacMan pacman = new NoisyPacMan();
		Param.read_Param();
		Param.actionSize = NUM_ACTS;
		Param.observationSize = NUM_OBS;
		PSRPlanningExperiment experiment = new PSRPlanningExperiment("PSRConfigs/pacman", "PlanningConfigs/pacman",pacman);
		experiment.runExperiment();
		experiment.publishResults("ResultPacMan");
	}

	public NoisyPacMan() {
		// TODO Auto-generated constructor stub
		super();
	}
	
	public NoisyPacMan(int maxRunLength)
	{
		super(maxRunLength);
	}
	@Override
	protected Observation getCurrentObservation(Random rando)
	{
		boolean[] lObsInfo = new boolean[16];
		int lYPos = pacManPos[0];
		int lXPos = pacManPos[1];

		/*check for walls*/

		//north
		if(gameMap[lYPos-1][lXPos] == 'x')
		{
			lObsInfo[0] = true;
		}
		//east
		if(gameMap[lYPos][lXPos+1] == 'x')
		{
			lObsInfo[1] = true;
		}
		//south
		if(gameMap[lYPos+1][lXPos] == 'x')
		{
			lObsInfo[2] = true;
		}
		//west
		if(gameMap[lYPos][lXPos-1] == 'x')
		{
			lObsInfo[3] = true;
		}
		
		for(int i = 0; i < 4; i++)
		{
			if(rando.nextDouble() < NOISE)
			{
				lObsInfo[i] = false;
			}
		}

		/*check for food smell*/
		int lFoodManhattanDis = computeFoodManhattanDist();

		if(lFoodManhattanDis <= 1)
		{
			lObsInfo[4] = true;
		}
		else if(lFoodManhattanDis <= 2)
		{
			lObsInfo[5] = true;
		}
		else if(lFoodManhattanDis <= 3)
		{
			lObsInfo[6] = true;
		}
		else if(lFoodManhattanDis <= 4)
		{
			lObsInfo[7] = true;
		}

		/*Check see ghosts or food*/
		//check north
		int[] tempLoc = pacManPos.clone();
		tempLoc[0]--;
		while(tempLoc[0] > 0 && tempLoc[1] < gameMap[0].length
				&& gameMap[tempLoc[0]][tempLoc[1]] != 'x')
		{

			for(int lGhost = 0; lGhost < ghostPoses.length; lGhost++)
			{
				if(tempLoc[0] == ghostPoses[lGhost][0] && tempLoc[1] == ghostPoses[lGhost][1])
				{
					lObsInfo[8] = true;
					break;
				}
			}
			if(lObsInfo[8] == true)
				break;
			tempLoc[0]--;

		}

		//check east
		tempLoc = pacManPos.clone();
		tempLoc[1]++;
		while(tempLoc[1] < gameMap[0].length -1
				&& gameMap[tempLoc[0]][tempLoc[1]] != 'x')
		{
			for(int lGhost = 0; lGhost < ghostPoses.length; lGhost++)
			{
				if(tempLoc[0] == ghostPoses[lGhost][0] && tempLoc[1] == ghostPoses[lGhost][1])
				{
					lObsInfo[9] = true;
					break;
				}
			}
			if(lObsInfo[9] == true)
				break;
			tempLoc[1]++;

		}

		//check south
		tempLoc = pacManPos.clone();

		tempLoc[0]++;
		while(tempLoc[0] < gameMap.length - 1
				&& gameMap[tempLoc[0]][tempLoc[1]] != 'x')		
		{
			for(int lGhost = 0; lGhost < ghostPoses.length; lGhost++)
			{
				if(tempLoc[0] == ghostPoses[lGhost][0] && tempLoc[1] == ghostPoses[lGhost][1])
				{
					lObsInfo[10] = true;
					break;
				}
			}
			if(lObsInfo[10] == true)
				break;
			tempLoc[0]++;
		}

		//check west
		tempLoc = pacManPos.clone();
		tempLoc[1]--;
		while( tempLoc[1] > 0
				&& gameMap[tempLoc[0]][tempLoc[1]] != 'x')		
		{
			for(int lGhost = 0; lGhost < ghostPoses.length; lGhost++)
			{
				if(tempLoc[0] == ghostPoses[lGhost][0] && tempLoc[1] == ghostPoses[lGhost][1])
				{
					lObsInfo[11] = true;
					break;
				}
			}
			if(lObsInfo[11] == true)
				break;
			tempLoc[1]--;
		}
		
		for(int i = 8; i < 12; i++)
		{
			if(rando.nextDouble() < NOISE)
			{
				lObsInfo[i] = false;
			}
		}

		lObsInfo[11] = powerPillCounter >= 0;

		return Observation.GetObservation(computeIntFromBinary(lObsInfo), currImmediateReward);
	}
	
	@Override
	protected int getNumberOfObservations() 
	{
		return NUM_OBS;
	}

	@Override
	public String getName()
	{
		return "NPacMan";
	}

}
