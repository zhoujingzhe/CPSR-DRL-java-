
package cpsr.planning;

import java.io.Serializable;
import java.util.ArrayList;

import org.apache.commons.lang3.SerializationUtils;
import org.jblas.util.Random;

import cpsr.environment.DataSet;
import cpsr.environment.TrainingDataSet;
import cpsr.environment.components.Action;
import cpsr.environment.components.ActionObservation;
import cpsr.model.APSR;
import cpsr.planning.ertapprox.actionensembles.ActionERTQPlanner;
import cpsr.planning.exceptions.PSRPlanningException;

public abstract class APSRPlanner implements IPlanner, Serializable
{
	@Override
	public APSRPlanner clone() throws CloneNotSupportedException {
		// TODO Auto-generated method stub
		APSRPlanner pclone = new ActionERTQPlanner(this.psr.clone(), this.qFunction);
		return pclone;
	}
	/**
	 * 
	 */
	private static final long serialVersionUID = 7322143475790810576L;
	/**The PSR associated with this planner*/
	protected APSR psr;

	public APSR get_psr()
	{
		return psr;
	}
	/**The Q function used for planning*/
	protected IQFunction qFunction;

	public IQFunction getqFunction()
	{
		return qFunction;
	}
	protected TrainingDataSet data;
	
	/**
	 * Constructs a planning object without any intialization.
	 * Plan must be learnt before getAction() and update() methods
	 * can be used.
	 */
	public APSRPlanner()
	{
		super();
	}

	public APSRPlanner(APSR psr, TrainingDataSet data)
	{
		this.psr = psr;
		this.data = data;
	}
	
	public APSRPlanner(APSR psr, TrainingDataSet data, IQFunction qFunction)
	{
		this.psr = psr;
		this.data = data;
		this.qFunction = qFunction;
	}
	@Override
	public Action getAction() throws Exception
	{
		if(qFunction == null)
		{
			throw new PSRPlanningException("Must learn a policy before asking for" +
					" best action!");
		}
		
		Action bestAction = null;
		double bestReward = Double.NEGATIVE_INFINITY;

		for(Action act : psr.getActionSet())
		{
			
			double currentReward = qFunction.getQValue(psr, act);
			if(currentReward > bestReward)
			{
				bestReward = currentReward;
				bestAction = act;
			}
		}
//		return (Action) bestAction.clone();
		return bestAction;
	}


	/**
	 * Updates the state with action-observation pair.
	 * 
	 * @param actob Action-observation pair.
	 * @throws Exception 
	 */
	public boolean update(ActionObservation actob) throws Exception
	{
		return psr.update(actob);
	}

	/**
	 * Resets the PSR to its start state.
	 */
	public void resetToStartState()
	{
		psr.resetToStartState();
	}

	/**
	 * @return Reference to current dataset in use.
	 */
	public DataSet getCurrentData()
	{
		return data;
	}
	
//	public void RemoveData()
//	{
//		this.data = null;
//	}
	/**
	 * Returns a QFunction learned using specified PSR, DataSet and tree ensemble parameters.  
	 * 
	 * @param runs Number of training runs to use when collecting intial data.
	 * @param iterations Number of iterations to use when training trees.
	 * @param treesPerEnsemble Number of trees per ensemble.
	 * @param k Number of splits to create at each inner node.  If k is null, then sqrt(number of attribute) will be used.
	 * @param nMin Specifies when trees stop growing if |set| < nMin at leaf, growing stops.
	 * @param pDiscount the discount factor.
	 * @return Q function policy
	 * @throws Exception 
	 */
	public abstract IQFunction learnQFunction(TrainingDataSet data, int runs, int iterations, int treesPerEnsemble, int k, int nMin, double pDiscount, int i) throws Exception;


}
