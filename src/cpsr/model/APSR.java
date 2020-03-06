package cpsr.model;

////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//import java.io.FileNotFoundException;
//import java.io.FileReader;
////////////////////////////////////////////
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.jblas.DoubleMatrix;
import org.jblas.exceptions.SizeException;
import org.json.simple.parser.ParseException;

import Parameter.Param;
import cpsr.environment.DataSet;
import cpsr.environment.TrainingDataSet;
import cpsr.environment.components.ActObSequenceSet;
import cpsr.environment.components.Action;
import cpsr.environment.components.ActionObservation;
import cpsr.environment.components.Observation;
import cpsr.model.components.Minf;
import cpsr.model.components.PredictionVector;
import cpsr.model.exceptions.PSRRuntimeException;
import cpsr.model.exceptions.PVException;
import cpsr.stats.PSRObserver;


@SuppressWarnings("serial")
public abstract class APSR implements Serializable, IPSR
{
	protected PredictionVector pv, initialPv;
	protected TrainingDataSet trainData;
	protected Minf mInf;
	protected boolean isBuilt;
	protected ActObSequenceSet tests, histories;
	protected Map<ActionObservation, DoubleMatrix> aoMats;
	public List<List<Double>> info;
	private DoubleMatrix randomVec;

	// clone a APSR model
	@Override
	public APSR clone()
	{
		APSR psr = new APSR() {
			@Override
			public void addPSRObserver(PSRObserver observer) {
				// TODO Auto-generated method stub
			}
			@Override
			public void update_traindata(TrainingDataSet trainData2) {
				// TODO Auto-generated method stub
			}
			@Override
			protected void performUpdate() {
				// TODO Auto-generated method stub
			}
			@Override
			protected void performBuild() throws Exception {
				// TODO Auto-generated method stub
			}
			@Override
			public void loadingExistPSR(String pSRPath) throws IOException, ParseException {
				// TODO Auto-generated method stub
			}
		};
		psr.mInf = this.mInf;
		psr.aoMats = this.aoMats;
		psr.histories =  this.histories;
		psr.tests = this.tests;
		psr.trainData = this.trainData;
		psr.initialPv = this.initialPv;
		psr.info = new ArrayList<List<Double>>();
		try {
			psr.pv = this.pv.clone();
		} catch (CloneNotSupportedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		psr.randomVec = this.randomVec;
		return psr;
	}
	
	public double getlikelihoods(PredictionVector pv, ActionObservation actob) throws Exception
	{
		DoubleMatrix tempMao = aoMats.get(actob);
		DoubleMatrix t = mInf.getVector().transpose().mmul(tempMao).mmul(pv.getVector());
		if (t.rows!=1||t.columns!=1)
		{
			throw new Exception("The output of prediction dimension has error!");
		}
		return t.get(0, 0);
	}
	
	public Map<ActionObservation, Double> getAllPrediction(PredictionVector pv, Action act)
	{
		Map<ActionObservation, Double> actobPreds = new HashMap<ActionObservation,Double>();
		HashSet<ActionObservation> actobs = trainData.getValidActionObservationSet();
		double sum = 0;
		for (ActionObservation actob :actobs)
		{
			if (act.getID() != actob.getAction().getID())
			{
				continue;
			}
			double Prob = 0;
			try {
				Prob = getlikelihoods(pv, actob);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			if (Prob < 0)
			{
				Prob = 0;
			}
			actobPreds.put(actob, Prob);
			sum += Prob;
		}
		for (ActionObservation actob : actobPreds.keySet())
		{
			double Prob = actobPreds.get(actob);
			actobPreds.put(actob, Math.round((Prob / sum) * 100000.0) / 100000.0);
		}
		return actobPreds;
	}

	private static final long serialVersionUID = -7364235618560975152L;

	/*
	 * a0o0r0, a0o0r1, a0o0r2, a1o0r0, a3o0r3, a3o1r3
	 */
	public DoubleMatrix getCoreTestProbability()
	{
		String[] coretests = {"a0o0r0", "a0o0r1", "a0o0r2", "a1o0r0", "a3o0r3", "a3o1r3"};
		double[] evalcoretests = new double[coretests.length];
		for (int index = 0; index < coretests.length; index++)
		{
			String test = coretests[index];
			int aid = test.indexOf("a");
			int a_id = Integer.parseInt(test.substring(aid + 1, aid + 2));
			int oid = test.indexOf("o");
			int o_id = Integer.parseInt(test.substring(oid + 1, oid + 2));
			int rid = test.indexOf("r");
			int r_id = Integer.parseInt(test.substring(rid + 1, rid + 2));
			Action act = Action.GetAction(a_id);
			Observation ob = Observation.GetObservation(o_id, r_id);
			ActionObservation actob = ActionObservation.getActionObservation(act, ob);
			DoubleMatrix Mao = this.aoMats.get(actob);
			DoubleMatrix temp_pv = this.getPredictionVector().getVector();
			DoubleMatrix temp_minf = this.mInf.getVector();
//			System.out.println("minf at "+Integer.toString(index)+":" + temp_minf);
			double prob = temp_minf.transpose().mmul(Mao.mmul(temp_pv)).get(0, 0);
			evalcoretests[index] = prob;
		}
		return new DoubleMatrix(evalcoretests);
	}
	
	
	/**
	 * Constructor initializes PSR of specified psrType with DataSet 
	 * 
	 * @param data The DataSet which the PSR is used to model.
	 * @param maxHistoryLength Max history length.
	 */
	protected APSR(TrainingDataSet data)
	{
		this.trainData = data;
		this.tests = data.getTests();
		this.histories = data.getHistories();
		info = new ArrayList<List<Double>>();
	}
	
	public APSR()
	{
	}
	// Building the PSR model
	@Override
	public void build(int svdDim, int seed) throws Exception
	{
		trainData.resetData();
		performBuild();
		isBuilt = true;
		if (Param.StartingAtSpecialPosition)
		{
			if (Param.GameName.equals("shuttle"))
			{
				update(ActionObservation.getActionObservation(Action.GetAction(2), Observation.GetObservation(2, -1)));
			}
			else if (Param.GameName.equals("niceEnv"))
			{
				update(ActionObservation.getActionObservation(Action.GetAction(1), Observation.GetObservation(2, -1)));
			}
			else if (Param.GameName.equals("Stand_tiger"))
			{
				update(ActionObservation.getActionObservation(Action.GetAction(1), Observation.GetObservation(0, 2)));
			}
			else if (Param.GameName.equals("Tiger95"))
			{
				update(ActionObservation.getActionObservation(Action.GetAction(0), Observation.GetObservation(0, 0)));
			}
			else if (Param.GameName.equals("Maze"))
			{
//				update(ActionObservation.getActionObservation(Action.GetAction(2), Observation.GetObservation(4, -1)));
				update(ActionObservation.getActionObservation(Action.GetAction(4), Observation.GetObservation(0, -1)));
			}
			else if (Param.GameName.equals("Stand_tiger_Test"))
			{
				update(ActionObservation.getActionObservation(Action.GetAction(1), Observation.GetObservation(0, 2)));
			}
		}
		initialPv = PredictionVector.BuildPredctiveVector(pv.getVector());
		// keep the same with SVD_Projection Dimension
		double[] randomArray = new double[svdDim];
		Random generator = new Random(Param.getRandomSeed());
		for (int i=0; i < svdDim; i++)
		{
			randomArray[i] = generator.nextDouble()*4-2;
		}
		this.randomVec = new DoubleMatrix(randomArray);
	}
	
	protected abstract void performBuild() throws Exception;

	/* (non-Javadoc)
	 * @see cpsr.model.IPSR#isBuilt()
	 */
	@Override
	public boolean isBuilt()
	{
		return isBuilt;
	}

	/* (non-Javadoc)
	 * @see cpsr.model.IPSR#getDataSet()
	 */
	@Override
	public DataSet getDataSet()
	{
		return trainData;
	}
	
	// Update the PSR model
	public void update()
	{
		trainData.resetData();
		performUpdate();
		initialPv = PredictionVector.BuildPredctiveVector(pv.getVector());
	}
	
	protected abstract void performUpdate();
	
	/*
	 * Update P(Q|h) to P(Q|hao)
	 */
	@Override
	public boolean update(ActionObservation ao) throws Exception
	{
		DoubleMatrix tempMao = aoMats.get(ao);
		DoubleMatrix numerator = null, denominator = null;
		
		try
		{
			numerator = (tempMao.mmul(pv.getVector()));
			denominator = (mInf.getVector().transpose()).mmul(numerator);
			if (denominator.get(0, 0) > 5)
			{
				System.err.println("The denominator is" + Double.toString(denominator.get(0, 0))); 
			}
			if (denominator.get(0, 0) == 0.0)
			{
				System.err.println("The denominator is zeros");
				throw new NullPointerException();
			}
		}
		catch(NullPointerException ex)
		{
			// if the observation is an unseen observation, treating it as an obscure sign.
			PredictionVector new_pv = null;
			Action a = ao.getAction();
			System.err.println("The observation" + ao + "are unseen before!");
			for (ActionObservation alreadyExisted_ao: aoMats.keySet())
			{
				if (!alreadyExisted_ao.getAction().equals(a))
				{
					continue;
				}
				PredictionVector pv_i = null;
				if (Param.update_pv.equals("PV"))
				{
					 pv_i = this.get_pv(alreadyExisted_ao);
				}
				else
				{
					pv_i = this.get_numerator(alreadyExisted_ao);
				}
				if (new_pv == null)
				{
					new_pv = pv_i;
				}
				else
				{
					new_pv = PredictionVector.Add(new_pv, pv_i);
				}
			}
			this.pv = new_pv;
			return false;
		} catch(SizeException ex)
		{
			System.err.println("The TempMao rows:" + Integer.toString(tempMao.rows) + ", columns:" + Integer.toString(tempMao.columns));
			System.err.println("The randomVec rows" + Integer.toString(randomVec.rows) + ", columns:" + Integer.toString(randomVec.columns));
			System.err.println("The pv rows" + Integer.toString(pv.getVector().rows) + ", columns:" + Integer.toString(pv.getVector().columns));
		}
		try 
		{
			if(!(denominator.get(0, 0) == 0.0)) 
			{
				pv = PredictionVector.BuildPredctiveVector(numerator.mul((1.0/denominator.get(0,0))));
			}
		} catch (PVException e) 
		{	
			e.printStackTrace();
			System.exit(0);
		}
		return false;
	}
	
	// return P(Q|hao), not modified the current P(Q|h)
	public PredictionVector get_pv(ActionObservation ao)
	{
		DoubleMatrix tempMao = aoMats.get(ao);
		DoubleMatrix numerator = null, denominator;
		PredictionVector new_pv = null;
		try
		{
			numerator = (tempMao.mmul(pv.getVector()));
		}
		catch(NullPointerException ex)
		{
			System.err.println("Unknown observation: no update performed in get_pv function");
		}
		denominator = (mInf.getVector().transpose()).mmul(numerator);
		try 
		{
			if(!(denominator.get(0, 0) == 0.0)) 
			{
				new_pv = PredictionVector.BuildPredctiveVector(numerator.mul((1.0/denominator.get(0,0))));
			}
			else
			{
				System.err.println("The denominator is zero");
			}
		} catch (PVException e) 
		{	
			e.printStackTrace();
			System.exit(0);
		}
		return new_pv;
	}
	// return P(aoQ|h), not modified the current P(Q|h)
	public PredictionVector get_numerator(ActionObservation ao)
	{
		DoubleMatrix tempMao = aoMats.get(ao);
		DoubleMatrix numerator = null;
		PredictionVector new_pv = null;
		try
		{
			numerator = (tempMao.mmul(pv.getVector()));
		}
		catch(NullPointerException ex)
		{
			//System.out.println(ao);
			System.err.println("Unknown observation: no update performed in get_pv function");
		}
		new_pv = PredictionVector.BuildPredctiveVector(numerator);
		return new_pv;
	}
	
	/* (non-Javadoc)
	 * @see cpsr.model.IPSR#resetToStartState()
	 */
	@Override
	public void resetToStartState()
	{
		if (initialPv == null)
		{
			System.err.println("The Initial PV is null in resetToStartState");
		}
		try
		{
			pv = PredictionVector.BuildPredctiveVector(initialPv.getVector().dup());
		}
		catch(PVException ex)
		{
			ex.printStackTrace();
		}
	}

	/* (non-Javadoc)
	 * @see cpsr.model.IPSR#getActionSet()
	 */
	@Override
	public HashSet<Action> getActionSet()
	{
		return this.trainData.getActionSet();
	}

	/**
	 * Returns deep copy of prediction vector.
	 * 
	 * @return Deep copy of prediction vector.
	 */
	public PredictionVector getPredictionVector()
	{
		try
		{
			return PredictionVector.BuildPredctiveVector(pv.getVector());
		}
		catch(PVException ex)
		{
			ex.printStackTrace();
		}
		return null;
	}

	/**
	 * Returns reference to Minf parameter.
	 * 
	 * @return Reference to Minf parameter.
	 */
	public Minf getMinf()
	{
		return mInf;
	}

	/**
	 * Returns reference to specified Mao parameter matrix. 
	 * 
	 * @param actob The action observation pair used to specify the Mao.
	 * @return The specified Mao parameter matrix. 
	 * @throws PSRRuntimeException
	 */
	public DoubleMatrix getAOMat(ActionObservation actob) throws PSRRuntimeException
	{
		DoubleMatrix mao = aoMats.get(actob);

		if(mao == null)
		{
			throw new PSRRuntimeException("There is no parameter associated with this action-observation pair");
		}
		else
		{
			return mao;
		}

	}

	/**
	 * Returns hash map of Mao parameters. 
	 * 
	 * @return Hash map of Mao parameters. 
	 */
	public Map<ActionObservation, DoubleMatrix> getAOMats()
	{
		return aoMats;
	}

	public abstract void loadingExistPSR(String pSRPath) throws IOException, ParseException;


	public abstract void update_traindata(TrainingDataSet trainData2);

}







