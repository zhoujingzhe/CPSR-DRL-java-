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

package cpsr.environment.components;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import cpsr.environment.DataSet;
import cpsr.environment.exceptions.EnvironmentException;
import Parameter.Param;

@SuppressWarnings("serial")
public class Observation implements Serializable {
	private static final long serialVersionUID = -9164516309249556193L;
	private final static Observation[] all_Observation;
	// extra position is for reset action
	static {
		if(Param.introducedReward)
		{
			all_Observation = new Observation[Param.observationSize*Param.rewardSize + 1];
			maxID = Param.rewardSize * Param.observationSize;
		}
		else
		{
			all_Observation = new Observation[Param.observationSize + 1];
			maxID = Param.observationSize;
		}
	}
	public static final Map<Double, Integer> RewardList = new HashMap<Double, Integer>();
	static {
		if (Param.GameName.equals("Stand_tiger")||Param.GameName.equals("Stand_tiger_Test"))
		{
			RewardList.put(-1.0, 0);
			RewardList.put(-100.0, 1);
			RewardList.put(30.0, 2);
			RewardList.put(-1000.0, 3);
		}
		else if (Param.GameName.equals("Tiger95"))
		{
			RewardList.put(-100.0, 0);
			RewardList.put(10.0, 1);
			RewardList.put(-1.0, 2);
		}
	}
	
	/**
	 * @serialField
	 * @deprecated
	 */
	protected DataSet dataSet;
	
	/**
	 * @serialField
	 */
	protected int oid;
	protected int rid;
	protected static int maxID;
	
	/**
	 * Default constructor. DO NOT USE!.
	 */
	protected Observation()
	{
		super();
	}
	
	/**
	 * Constructs Observation using only id.
	 * Use this constructor if adding observation to data set
	 * or if one plans on setting the DataSet at a later time.
	 * 
	 * @param id The action id.
	 */
	private Observation(int oid, int rid)
	{
		this.oid = oid;
		if (Param.introducedReward)
		{
			this.rid = rid;
		}
		else
		{
			this.rid = -1;
		}
	}
	
	/**
	 * Constructs observation associated with a particular
	 * DataSet using specified integer id.
	 * 
	 * @param id
	 * @param dataSet
	 * @deprecated
	 */
	public Observation(int oid, double rid, DataSet dataSet)
	{
		this.oid = oid;
		this.rid = RewardList.get(rid);
		this.dataSet = dataSet;
	}
	
	/**
	 * Returns a binary string representing observation with length equal
	 * to the max length specified by max observation ID of DataSet.
	 * 
	 * @return Binary representation of observation. 
	 * @throws EnvironmentException
	 */
//	public String toBinaryString() throws EnvironmentException
//	{
//		int targetLength = (Integer.toBinaryString(maxID)).length();
//		String binaryRep = Integer.toBinaryString(this.getID());
//
//		if(targetLength < binaryRep.length())
//		{
//			throw new EnvironmentException("Integer ID of observation exceeds maximum specifed by environment");
//		}
//		else if(targetLength > binaryRep.length())
//		{
//			int binarylength = targetLength - binaryRep.length();
//			for(int i = 0; i < binarylength; i++)
//			{
//				binaryRep =  "0" + binaryRep;
//			}
//		}
//		return binaryRep;
//	}
	
	/**
	 * Returns DataSet associated with this observation.
	 * 
	 * @return DataSet associated with this observation. 
	 * @deprecated
	 */
	public DataSet getDataSet()
	{
		try
		{
			return dataSet;
		}
		catch(NullPointerException ex)
		{
			throw new EnvironmentException("Child classes of Observation must explicitly set dataSet field in constructor, or" +
					"by using setDataSet(DataSet) method");
		}
	}
	
	
	public int getID()
	{
		if (Param.introducedReward)
		{
			return getoID() * Param.rewardSize + getrID();
		}
		else
		{
			return getoID();
		}
	}
	
	/**
	 * Return unique identifying integer ID for observation.
	 * 
	 * @return Unique identifying integer ID for observation
	 */
	public int getoID()
	{
		try
		{
			return oid;
		}
		catch(NullPointerException ex)
		{
			throw new EnvironmentException("Child classes of Observation must explicitly set id field in constructor");
		}
	}
	
	public int getrID()
	{
		try
		{
			return rid;
		}
		catch(NullPointerException ex)
		{
			throw new EnvironmentException("Child classes of Observation must explicitly set id field in constructor");
		}
	}
	/**
	 * Sets the data set.
	 */
	public void setData(DataSet dataSet)
	{
		this.dataSet = dataSet;
	}
	
	@Override
	public int hashCode()
	{
		if (Param.introducedReward)
		{
			return Param.rewardSize * getoID() + getrID();
		}
		else
		{
			return getoID();
		}
	}
	
	/**
	 * Tests for equality 
	 * 
	 * @param ob The observation that is being compared
	 * @return True if both obsevations equal.
	 */
	@Override
	public boolean equals(Object ob)
	{
		if (Param.introducedReward)
		{
			return  equalsForObservation(ob) && equalsForRewards(ob);
		}
		else
		{
			return equalsForObservation(ob);
		}
	}
	
	public boolean equalsForObservation(Object ob)
	{
		return this.getoID() == ((Observation)ob).getoID();
	}
	
	public boolean equalsForRewards(Object ob)
	{
		return this.getrID() == ((Observation)ob).getrID();
	}
	
	////////////////////////////////////////////////////////////////////////////
	/// Changed by ZJZ
	
	public static Observation GetObservation(int o_id, double reward)
	{
		int id;
		int rid = -1;
		if (Param.introducedReward)
		{
			rid = RewardList.get(reward);	
			id = o_id * Param.rewardSize + rid;
		}
		else
		{
			 id = o_id;
		}
		if (all_Observation[id] == null)
		{
			synchronized (all_Observation) {
				all_Observation[id] = new Observation(o_id, rid);
			}
		}
		return all_Observation[id];
	}
	
	public static Observation GetObservation(int o_id, int r_id)
	{
		int rid = -1;
		int id;
		if (Param.introducedReward)
		{
			rid = r_id;
			id = o_id * Param.rewardSize + rid;
		}
		else
		{
			id = o_id;
		}
		if (all_Observation[id] == null)
		{
			synchronized (all_Observation) {
				all_Observation[id] = new Observation(o_id, r_id);
			}
		}
		return all_Observation[id];
	}
	
	@Override
	public String toString() throws EnvironmentException
	{
		String info;
		if (Param.introducedReward)
		{
			info = "oid" + Integer.toString(this.oid) + "rid" + Integer.toString(this.rid);
		}
		else
		{
			info = "oid" + Integer.toString(this.oid);
		}
		return info;
	}
}
