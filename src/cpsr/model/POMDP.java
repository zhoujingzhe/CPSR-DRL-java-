package cpsr.model;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.jblas.DoubleMatrix;

import cpsr.environment.components.Action;
import cpsr.environment.components.ActionObservation;
import cpsr.environment.components.Observation;

public class POMDP
{
	int[] ExpectedObservation;
	Map<Integer, int[]> policy = new HashMap<Integer, int[]>();
	Map<Integer, double[][]> Ts;
	Map<Integer, Map<Integer,double[][]>> Os;
	int obSize;
	public void setTs(Map<Integer, double[][]> t)
	{
		Ts = t;
	}
	public void setOs(Map<Integer, Map<Integer,double[][]>> o)
	{
		Os = o;
	}
	public void setbelief(double[] belief)
	{
		initialbelief = belief;
		this.belief = initialbelief.clone();
	}
	public void reset()
	{
		belief = initialbelief.clone();
	}
	double[] belief;
	double[] initialbelief;
	int states;
	Map<Integer, List<double[]>> valueFunctions = new HashMap<Integer, List<double[]>>();
	public POMDP(String Path, String Path1, int num_states, int num_obs) throws Exception
	{
		states = num_states;
		obSize = num_obs;
		readValueFunctions(Path);
//		readPolicy(Path1);
	}
	private void readPolicy(String file) throws IOException
	{
		try(BufferedReader br = new BufferedReader(new FileReader(file))) {
			for(String line; (line = br.readLine()) != null; ) {
		        String[] nums = line.split(" ");
		        int id = Integer.parseInt(nums[0]);
		        
		        int[] n = new int[obSize + 1];
		        int idx = 0;
		        for (int i = 0; i < nums.length - 1; i++)
		        {
		        	if (!nums[i + 1].isEmpty())
		        	{
		        		n[idx++] = Integer.parseInt(nums[i+1]);
		        	}
		        }
		        policy.put(id, n);
			}
		}
	}
	private void readValueFunctions(String file) throws Exception
	{
		int count = 0;
		try(BufferedReader br = new BufferedReader(new FileReader(file))) {
		    for(String line; (line = br.readLine()) != null; ) {
		        if (!line.contains(" "))
		        {
//		        	int actid = Integer.parseInt(line.split(":")[0]);
		        	int actid = Integer.parseInt(line);
		        	line = br.readLine();
		        	String[] lines = line.split(" ");
		        	if (lines.length != states)
		        	{
		        		throw new Exception("the belief state's length is not equal to coefficient vectors!");
		        	}
		        	double[] values = new double[lines.length + 1];
		        	int idx = 0;
		        	for (String co:lines)
		        	{
		        		values[idx++] = Double.parseDouble(co);
		        	}
		        	values[idx] = count++;
		        	if (!valueFunctions.containsKey(actid))
		        	{
		        		valueFunctions.put(actid, new ArrayList<double[]>());
		        	}
	        		valueFunctions.get(actid).add(values);
	        		line = br.readLine();
		        }
		    }
		}
	}
	public void updateBelief(ActionObservation actob)
	{
		int actid = actob.getAction().getID();
		int obid = actob.getObservation().getoID();
		double[][] Ta = Ts.get(actid);
		double[][] Oa = Os.get(actid).get(obid);
		DoubleMatrix T = new DoubleMatrix(Ta);
		DoubleMatrix O = new DoubleMatrix(Oa);
		DoubleMatrix Belief = new DoubleMatrix(belief).transpose();
		DoubleMatrix numerator = Belief.mmul(T).mmul(O);
		double denumerator = numerator.mmul(DoubleMatrix.ones(states, 1)).get(0, 0);
		belief = numerator.mul(1.0 / denumerator).toArray();
	}
	
	public int StartingPolicyGraph() throws Exception
	{
		int bestAction = -1;
		int Bestid = -1;
		double MaxValue = -Double.MAX_VALUE;
		for (int actid = 0; actid < valueFunctions.size(); actid++)
		{
			List<double[]> values  = valueFunctions.get(actid);
			for (double[] value1:values)
			{
				if (value1.length - 1 != belief.length)
				{
					throw new Exception("the length of belief vectors are not equal to value vector!");
				}
				double val = 0;
				int i = 0;
				for (; i < belief.length; i++)
				{
					val += (value1[i] * belief[i]);
				}
				if (val > MaxValue)
				{
					bestAction = actid;
					MaxValue = val;
					Bestid = (int) value1[i];
				}
			}
		}
		int[] t = policy.get(Bestid);
		if (t[0] != bestAction)
		{
			throw new Exception("The value function's output is different with policy!");
		}
		ExpectedObservation = t;
		return bestAction;
	}
	public int getActionByPolicy(int Oid)
	{
		int Pid = ExpectedObservation[Oid + 1];
		int[] h = policy.get(Pid);
		ExpectedObservation = h;
		return h[0];
	}
	public int getAction() throws Exception {
		int bestAction = -1;
		double MaxValue = -Double.MAX_VALUE;
		for (int actid = 0; actid < valueFunctions.size(); actid++)
		{
			List<double[]> values  = valueFunctions.get(actid);
			for (double[] value1:values)
			{
				if (value1.length - 1 != belief.length)
				{
					throw new Exception("the length of belief vectors are not equal to value vector!");
				}
				double val = 0;
				for (int i = 0; i < belief.length; i++)
				{
					val += (value1[i] * belief[i]);
				}
				if (val > MaxValue)
				{
					bestAction = actid;
					MaxValue = val;
				}
			}
		}
		return bestAction;
	}
}
