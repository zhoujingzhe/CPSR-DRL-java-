
package cpsr.stats;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import Parameter.Param;
import cpsr.environment.TrainingDataSet;
import cpsr.environment.components.Action;
import cpsr.environment.components.ActionObservation;
import cpsr.environment.components.Observation;
import cpsr.environment.simulation.domains.Maze;
import cpsr.environment.simulation.domains.Stand_tiger;
import cpsr.environment.simulation.domains.Stand_tiger_Test;
import cpsr.environment.simulation.domains.Tiger95;
import cpsr.environment.simulation.domains.niceEnv;
import cpsr.environment.simulation.domains.shuttle;
import cpsr.model.APSR;
import cpsr.model.IPredictor;

public class Likelihood 
{
	
	APSR aPsr;
	TrainingDataSet aData;
	IPredictor aPredictor;
	
	public void PrintOutLikelihoodsOfAllao(List<ActionObservation> Listao, int policyIters, String FileName, String GameName) throws IOException
	{
		File file = new File("Likelihoods/PI" + Integer.toString(policyIters));
		file.mkdirs();
		BufferedWriter write = new BufferedWriter(new FileWriter("Likelihoods/PI" + Integer.toString(policyIters) + "/" + FileName));

		int num = 0;
		if (GameName.equals("Stand_tiger"))
		{
			num = Stand_tiger.Actions.length;
		}
		else if (GameName.equals("Tiger95"))
		{
			num = Tiger95.Actions.length;
		}
		else if (GameName.equals("niceEnv"))
		{
			num = niceEnv.Actions.length;
		}
		else if (GameName.equals("shuttle"))
		{
			num = shuttle.Actions.length;
		}
		else if (GameName.equals("Maze"))
		{
			num = Maze.Actions.length;
		}		
		else if (GameName.equals("Stand_tiger_Test"))
		{
			num = Stand_tiger_Test.Actions.length;
		}
		for (int actid = 0; actid < num; actid++)
		{
			Map<ActionObservation, Double> actPreds = aPsr.getAllPrediction(aPsr.getPredictionVector(), Action.GetAction(actid));
			for (ActionObservation actob: actPreds.keySet())
			{
				int aid = actob.getAction().getID();
				int oid = actob.getObservation().getoID();
				int rid = -1;
				double likelihood_ao = actPreds.get(actob);
				if (Param.introducedReward)
				{
					rid = actob.getObservation().getrID();
				}
				if (GameName.equals("Stand_tiger") && Param.introducedReward)
				{
					write.write(Stand_tiger.Actions[aid] + "  " + Stand_tiger.Observations[oid] + Stand_tiger.rewards[rid] + ", ");
				}
				else if (GameName.equals("Stand_tiger"))
				{
					write.write(Stand_tiger.Actions[aid] + "  " + Stand_tiger.Observations[oid] + ", ");
				}
				else if (GameName.equals("Tiger95") && Param.introducedReward)
				{
					write.write(Tiger95.Actions[aid] + "  " + Tiger95.Observations[oid] + " " + Tiger95.rewards[rid] + ", ");
				}
				else if (GameName.equals("Tiger95"))
				{
					write.write(Tiger95.Actions[aid] + "  " + Tiger95.Observations[oid] + ", ");
				}
				else if (GameName.equals("niceEnv"))
				{
					write.write(niceEnv.Actions[aid] + "  " + niceEnv.Observations[oid] + ", ");
				}
				else if (GameName.equals("shuttle"))
				{
					write.write(shuttle.Actions[aid] + "  " + shuttle.Observations[oid] + ", ");
				}
				else if (GameName.equals("Maze"))
				{
					write.write(Maze.Actions[aid] + "  " + Maze.Observations[oid] + ", ");
				}				
				else if (GameName.equals("Stand_tiger_Test") && Param.introducedReward)
				{
					write.write(Stand_tiger_Test.Actions[aid] + "  " + Stand_tiger_Test.Observations[oid] + " " + Stand_tiger_Test.rewards[rid] +", ");
				}
				write.write("The likelihoods of ao is " + Double.toString(likelihood_ao));
				write.write("\n");	
			}
		}
		write.close();
	}
	
	/**
	 * Constructs a likelihood object given psr model data and predictor.
	 * 
	 * @param pPsr A PSR model.
	 * @param pData A DataSet.
	 * @param pPredictor A prediction object defined over the PSR model.
	 */
	public Likelihood(APSR pPsr, TrainingDataSet pData, IPredictor pPredictor)
	{
		aPsr = pPsr;
		aData = pData;
		aPredictor = pPredictor;
	}

	/**
	 * @return The likelihood of data given the PSR model.
	 * @throws Exception 
	 */
	public double getLikelihoodOfData() throws Exception
	{
		double lLikelihood;
		
		lLikelihood = 0.0;
		for(int i = 0; i < aData.getNumberOfRunsInBatch(); i++)
		{
			double lRunLike = 1.0;
			boolean firstStep = true;
			while(true)
			{
				if(firstStep)
				{
					firstStep = false;
				}
				else
				{
					if(checkForReset()) break;
				}
				ActionObservation ao = aData.getNextActionObservationForPlanning();
				lRunLike*=aPredictor.getImmediateProb(ao.getAction(),
						ao.getObservation());
				aPsr.update(ao);
	
			}
			lLikelihood+=lRunLike;
		}
		lLikelihood = lLikelihood/((double)aData.getNumberOfRunsInBatch());
		return lLikelihood;
	}
	
	/**
	 * @return The likelihood of data given the PSR model.
	 * @throws Exception 
	 */
	public List<double[]> getKStepLikelihoodsOfData(int k) throws Exception
	{
		
		List<List<Double>> likehoods = new ArrayList<List<Double>>(); 
		
		for(int i = 0; i < k; i++)
		{
			likehoods.add(new ArrayList<Double>());
		}
		
		for(int i = 0; i < aData.getNumberOfRunsInBatch(); i++)
		{
			double lRunLike = 1.0;
			boolean firstStep = true;
			int iterCount = 0;
			while(true)
			{
				iterCount++;
				if(firstStep)
				{
					firstStep = false;
				}
				else
				{
					if(checkForReset() || iterCount > k) break;
				}
				ActionObservation ao = aData.getNextActionObservation();
				
				if(ao.getID() == -1)
					break;
				
				lRunLike*=aPredictor.getImmediateProb(ao.getAction(),
						ao.getObservation());
				likehoods.get(iterCount-1).add(lRunLike);
				aPsr.update(ao);
			}
		}
		
		List<double[]> results = new ArrayList<double[]>();
		for(List<Double> kLike : likehoods)
		{
			double[] res = new double[2];
			
			res[0] = Basic.mean(kLike);
			res[1] = Basic.std(kLike, res[0]);
			
			results.add(res);
		}
		
		return results;
	}
	
	/**
	 * @return The likelihood of the first observations given the PSR model.
	 * @throws Exception 
	 */
	public double getOneStepLikelihoodOfData() throws Exception
	{
		double lLikelihood;
		
		lLikelihood = 0.0;
		for(int i = 0; i < aData.getNumberOfRunsInBatch(); i++)
		{
				Action nextAct = aData.getNextActionObservationForPlanning().getAction();
				Observation nextObs = aData.getNextActionObservationForPlanning().getObservation();
				lLikelihood+=aPredictor.getImmediateProb(nextAct, nextObs);
				aPsr.update(ActionObservation.getActionObservation(nextAct, nextObs));
	
		}
		
		lLikelihood = lLikelihood/((double)aData.getNumberOfRunsInBatch());
		
		return lLikelihood;
	}
	
	/**
	 * Helper method determines if a run terminated.
	 * If so, true returned and prediction vector reset. 
	 * 
	 * @return Boolean representing whether reset performed.
	 */
	private boolean checkForReset()
	{
		if(aData.resetPerformed())
		{
			aPsr.resetToStartState();
			return true;
		}
		else
		{
			return false;
		}
	}
}
