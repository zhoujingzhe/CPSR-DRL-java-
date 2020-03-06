package cpsr.stats;

import java.util.ArrayList;
import java.util.List;

import Parameter.Param;

/**
 * Computes statistics relevant to planning.
 * 
 * @author William Hamilton
 */
public class PlanningStats
{

	public static double getAverageDiscountedReward(List<List<Double>> runRewards, double pDiscount)
	{
		double lAvReward = 0.0;
		double lRunReward;

		for(List<Double> lRewardVec : runRewards)
		{
			lRunReward = 0;
			for(int i = 0; i < lRewardVec.size(); i++)
			{
				if(i == 0)
				{
					lRunReward+=lRewardVec.get(i);
				}
				else
				{
					lRunReward+=Math.pow(pDiscount, (double)i)*lRewardVec.get(i);
				}	
			}
			lAvReward+=lRunReward;
		}
		return lAvReward/((double)runRewards.size());
	}
	
	public static double[] getAverageReward(List<List<Double>> runRewards, double pDiscount)
	{
		double lAvReward = 0.0;
		int count_win = 0;
		int count_lose = 0;
		int actioncount = 0;
		for(List<Double> lRewardVec : runRewards)
		{
			double lRunReward = 0;
			for(int i = 0; i < lRewardVec.size(); i++)
			{
				double reward = lRewardVec.get(i);
				lRunReward+=reward;
				actioncount++;
				if (reward == 30.0&&Param.GameName.equals("Stand_tiger"))
				{
					count_win++;
				}
				else if (reward == 10.0 && Param.GameName.equals("Tiger95"))
				{
					count_win++;
				}
				else if ((reward == -100.0 || reward == -1000.0)&&Param.GameName.equals("Stand_tiger"))
				{
					count_lose++;
				}
				else if (reward == -100.0 && Param.GameName.equals("Tiger95"))
				{
					count_lose++;
				}
				else if (reward == 100.0 && Param.GameName.equals("niceEnv"))
				{
					count_win++;
				}
				else if ((reward == -50.0 || reward == -10.0) && Param.GameName.equals("niceEnv"))
				{
					count_lose++;
				}
				else if (reward == 10.0 && Param.GameName.equals("shuttle"))
				{
					count_win++;
				}
				else if (reward == -3.0 && Param.GameName.equals("shuttle"))
				{
					count_lose++;
				}
				else if (reward == 10.0 && Param.GameName.equals("Maze"))
				{
					count_win++;
				}
				else if (reward == -100.0 && Param.GameName.equals("Maze"))
				{
					count_lose++;
				}
			}
			lAvReward+=lRunReward;
		}
		int size = runRewards.size();
		
		if (!Param.GameName.equals("PacMan")&&!Param.RunningVersion.equals("V1"))
		{
			size = (count_win + count_lose);
		}
		double winProb = (double)count_win / size;
		double[] ret = new double[2];
		ret[0] = winProb;
		ret[1] = lAvReward/(double)(runRewards.size());
		System.out.println("CountWin:" + Double.toString(count_win));
		System.out.println("CountLoss:" + Double.toString(count_lose));
		System.out.println("Size of actions:" + Integer.toString(actioncount));
		return ret;
	}

	public static double getVarOfReward(List<List<Double>> runRewards, double pAv)
	{		
		double lVar = 0.0;
		double lRunReward;

		for(List<Double> lRewardVec : runRewards)
		{
			lRunReward = 0;
			for(int i = 0; i < lRewardVec.size(); i++)
			{
				lRunReward+=lRewardVec.get(i);
			}
			lVar += Math.pow((lRunReward-pAv), 2);
		}
		return lVar/((double)runRewards.size());
	}
	
	public static double getVarOfDiscountedReward(List<List<Double>> runRewards, double pDiscount, double pAv)
	{		
		double lVar = 0.0;
		double lRunReward;

		for(List<Double> lRewardVec : runRewards)
		{
			lRunReward = 0;
			for(int i = 0; i < lRewardVec.size(); i++)
			{
				if(i == 0)
				{
					lRunReward+=lRewardVec.get(i);
				}
				else
				{
					lRunReward+=Math.pow(pDiscount, (double)i)*lRewardVec.get(i);
				}	
			}
			lVar += Math.pow((lRunReward-pAv), 2);
		}
		return lVar/((double)runRewards.size());
	}

	public static double getAverageLengthOfRun(List<List<Double>> runRewards)
	{
		ArrayList<Double> lLengthVec = new ArrayList<Double>();

		for(List<Double> pSingleRun : runRewards)
		{
			lLengthVec.add((double)pSingleRun.size());
		}

		return Basic.mean(lLengthVec);
	}
}
