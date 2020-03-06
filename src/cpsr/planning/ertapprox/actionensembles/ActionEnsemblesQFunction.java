/*
 *   Copyright 2012 William Hamilton
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */
package cpsr.planning.ertapprox.actionensembles;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;

import org.apache.commons.math.stat.StatUtils;

import Parameter.Param;
import cpsr.environment.components.Action;
import cpsr.model.APSR;
import cpsr.planning.IQFunction;
import cpsr.planning.exceptions.PSRPlanningException;


public class ActionEnsemblesQFunction implements IQFunction, Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = 5515640625421609280L;
	APSR psr;
	HashMap<Action, ActionERTEnsemble> erts;
	ArrayList<HashMap<Action, ActionERTEnsemble>> array_erts; 
	
	public HashMap<Action, ActionERTEnsemble> get_erts()
	{
		return erts;
	}
	
	public ArrayList<HashMap<Action, ActionERTEnsemble>> get_array_erts()
	{
		return array_erts;
	}
	
	/**
	 * Constructs an ActionEnsembleQFunction for specified DataSet and PSR.
	 * Method addActionEnsemble(ActionERTEnsemble) must be called and ensemble for each
	 * action must be added before using this object.
	 * 
	 * @param psr The associated PSR.
	 */
	public ActionEnsemblesQFunction(APSR psr)
	{
		this.psr = psr;
		this.erts = new HashMap<Action, ActionERTEnsemble>();
		this.array_erts = new ArrayList<HashMap<Action, ActionERTEnsemble>>();
	}
	
	/**
	 * Constructs an ActionEnsembleQFunction for PSR
	 * with specified collection of ActionERTEnsembles.
	 * 
	 * @param psr The associated PSR.
	 * @param actionEnsembles A complete mapping of actions to actionEnsembles
	 */
	public ActionEnsemblesQFunction(APSR psr, HashMap<Action, ActionERTEnsemble> actionEnsembles)
	{
		this.psr = psr;
		this.erts = actionEnsembles;
		if (this.erts.size() != Param.actionSize)
		{
			System.err.println("Error on FittedQ!");
		}
	}
	
	public ActionEnsemblesQFunction(APSR psr, ArrayList<HashMap<Action, ActionERTEnsemble>> Array_actionEnsembles)
	{
		this.psr = psr;
		this.array_erts = Array_actionEnsembles;
		if (array_erts.size()!=Param.num_atoms)
		{
			System.err.println("Error on distributions!");
		}
	}
	
	@Override
	public double getQValue(APSR psr, Action act) throws Exception
	{
		validateConstruction();
		if (Param.C51 || Param.Quantile_DRL)
		{
			return getZValue(psr, act);
		}
		return erts.get(act).getValueEstimate(psr.getPredictionVector());
	}
	
	public double getZValue(APSR psr, Action act) throws Exception
	{
		validateConstruction();
		double[] dist = new double[array_erts.size()];
		double uniform_loss = 1 / (array_erts.size());
		for (int index = 0; index < array_erts.size(); index++)
		{
			HashMap<Action, ActionERTEnsemble> ert = array_erts.get(index);
			double prob = ert.get(act).getValueEstimate(psr.getPredictionVector());
			
			dist[index] = (1-Param.tao_uniform_loss) * prob + Param.tao_uniform_loss * uniform_loss;
		}
		return ActionERTQPlanner.Score_of_Distribution(dist) - Param.variance_part * StatUtils.variance(dist);
	}
	/**
	 * Adds an action ensemble to the set of action ensembles.
	 * 
	 * @param actionEnsemble The single action ensemble to add.
	 */
	public void addActionEnsemble(ActionERTEnsemble actionEnsemble)
	{
		erts.put(actionEnsemble.getAssociatedAction(), actionEnsemble);
	}
	
	public void addActionEnsemble(ArrayList<ActionERTEnsemble> actionEnsemble)
	{
//		erts.put(actionEnsemble.getAssociatedAction(), actionEnsemble);
	}
	/**
	 * Tests whether object has been fully created.
	 */
	private void validateConstruction()
	{
		if (Param.C51 || Param.Quantile_DRL)
		{
			if (array_erts.get(0).keySet().size() != psr.getActionSet().size())
			{
				throw new PSRPlanningException("ActionEnsembleQFunction must have ensemble for each action." +
						"Either construct with full ensembles or add using addActionEnsemble(ActionERTEnsemble) function");
			}
		}
		else
		{
			if(erts.keySet().size() != psr.getActionSet().size())
			{
				throw new PSRPlanningException("ActionEnsembleQFunction must have ensemble for each action." +
						"Either construct with full ensembles or add using addActionEnsemble(ActionERTEnsemble) function");
			}
		}
		
	}

}
