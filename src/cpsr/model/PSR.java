package cpsr.model;




import java.util.HashMap;


import org.jblas.DoubleMatrix;

import cpsr.environment.TrainingDataSet;
import cpsr.environment.components.Action;
import cpsr.environment.components.ActionObservation;
import cpsr.model.components.Minf;
import cpsr.model.components.PredictionVector;

///////////////////////////////////////////////////////////////////////////////
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import com.google.gson.Gson;
import com.google.gson.JsonIOException;
import com.google.gson.JsonSyntaxException;
import cpsr.stats.PSRObserver;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import com.google.gson.reflect.TypeToken;

import Parameter.Param;
import cpsr.environment.components.Observation;


import java.lang.reflect.Type;
import java.util.Map;
//////////////////////////////////////////////////////////////////////////////////////

public class PSR extends APSR {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public PSR(TrainingDataSet data) {
		super(data);
		// TODO Auto-generated constructor stub
	}
	
	public PSR() {
		super();
	}

	@Override
	public void addPSRObserver(PSRObserver observer) {
		// TODO Auto-generated method stub

	}

	@Override
	protected void performBuild() {
		// TODO Auto-generated method stub

	}

	@Override
	protected void performUpdate() {
		// TODO Auto-generated method stub

	}
	
	public void loadingExistPSR(String path) throws IOException, ParseException {
		Gson gson = new Gson();
		//JSON parser object to parse read file
        JSONParser jsonParser = new JSONParser();
         
        try (FileReader reader = new FileReader(path))
        {
        	boolean isStand_Tiger = path.indexOf("StandTiger") !=-1? true: false;
        	boolean isTiger95 = path.indexOf("Tiger95") !=-1? true: false;
        	boolean isniceEnv = path.indexOf("NiceEnv") !=-1? true: false;
        	boolean isshuttle = path.indexOf("shuttle") !=-1? true: false;
        	boolean ismaze = path.indexOf("Maze") !=-1? true: false;
            //Read JSON file
        	JSONObject obj = (JSONObject)jsonParser.parse(reader);
        	String line2 = obj.get("pv").toString();
        	double[][] pv = gson.fromJson(line2, double[][].class);
        	DoubleMatrix pv_vector = new DoubleMatrix(pv);
        	this.initialPv = PredictionVector.BuildPredctiveVector(pv_vector);
        	this.pv = PredictionVector.BuildPredctiveVector(this.initialPv.getVector().dup());

        	String line = obj.get("m_inf").toString();
        	double[][][] matrix = gson.fromJson(line, double[][][].class);
        	DoubleMatrix m_inf = new DoubleMatrix(matrix[0]);
        	this.mInf = new Minf(m_inf);
        	
        	Type type = new TypeToken<Map<String, double[][]>>(){}.getType();
        	String line1 = obj.get("M_ao").toString();
        	Map<String, double[][]> M_ao = gson.fromJson(line1, type);
        	this.aoMats = new HashMap<ActionObservation, DoubleMatrix>();
        	for (String ao:M_ao.keySet())
        	{
        		DoubleMatrix M = new DoubleMatrix(M_ao.get(ao));
        		char[] ao_array = ao.toCharArray();
        		int a_id = Character.getNumericValue(ao_array[1]);
        		int o_id = Character.getNumericValue(ao_array[3]);
        		
        		double r_id = -1;
        		if(Param.introducedReward)
        		{
        			r_id = Character.getNumericValue(ao_array[5]);
        		}
        		int max_a = 0;
        		int max_o = 0;
        		if (isTiger95)
        		{
        			max_a = 3;
        			max_o = 2;
        		}
        		if (isStand_Tiger)
        		{
        			max_a = 5;
        			max_o = 6;
        		}
        		if (isniceEnv)
        		{
        			max_a = 5;
        			max_o = 5;
        		}
        		if (isshuttle)
        		{
        			max_a = 3;
        			max_o = 5;
        		}
        		if (ismaze)
        		{
        			max_a = 5;
        			max_o = 6;
        		}
        		if (max_a == 0 || max_o == 0)
        		{
        			System.out.println("there is no included in game store!");
        		}
        		Action a = Action.GetAction(a_id);
        		if (Param.introducedReward && isStand_Tiger)
        		{
	        		if (r_id == 0)
	        		{
	        			r_id = -100.0;
	        		}
	        		else if (r_id == 1)
	        		{
	        			r_id = 30.0;
	        		}
	        		else if (r_id == 2)
	        		{
	        			r_id = -1000.0;
	        		}
	        		else if (r_id == 3)
	        		{
	        			r_id = -1.0;
	        		}
        		}
        		else if (Param.introducedReward && isTiger95)
        		{
	        		if (r_id == 0)
	        		{
	        			r_id = -100.0;
	        		}
	        		else if (r_id == 1)
	        		{
	        			r_id = 10.0;
	        		}
	        		else if (r_id == 2)
	        		{
	        			r_id = -1.0;
	        		}
        		}
        		Observation o = Observation.GetObservation(o_id, r_id);
        		ActionObservation AO = ActionObservation.getActionObservation(a, o);
        		this.aoMats.put(AO, M);
        	}
		} catch (JsonIOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (JsonSyntaxException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	@Override
	public void update_traindata(TrainingDataSet trainData2) {
		// TODO Auto-generated method stub
		this.trainData = trainData2;
	}
}
