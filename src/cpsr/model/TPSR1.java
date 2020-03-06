package cpsr.model;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.jblas.DoubleMatrix;
import org.jblas.Singular;
import org.jblas.Solve;
import org.json.simple.parser.ParseException;
import Parameter.Param;
import cpsr.environment.TrainingDataSet;
import cpsr.environment.components.ActObSequenceSet;
import cpsr.environment.components.ActionObservation;
import cpsr.environment.components.IntSeq;
import cpsr.model.components.Minf;
import cpsr.model.components.PredictionVector;
import cpsr.stats.PSRObserver;
import jparfor.Functor;
import jparfor.MultiThreader;
import jparfor.Range;

public class TPSR1 extends APSR 
{
	private static final long serialVersionUID = 3166796096290306882L;

	public static final int UPPER_HIST_LEN_LIMIT = 1000000;
	public static final double STABILITY_CONSTANT = 1.0/1000.0;
	
	protected double minSingularVal;
	protected int maxDim, maxHistLen, maxTestLen;
	
	protected int aoMats_rows, aoMats_columns;
	
	protected DoubleMatrix hist, th, pseudoInverse;
	protected DoubleMatrix[] svdResults;
	
	protected List<PSRObserver> observers;
//	protected int resetCount;

	protected int[] CountNumberForTHMats = {0};
	protected int[] CountNumberForHMats = {0};
	protected Map<ActionObservation, Integer> CountNumberForC_ao = Collections.synchronizedMap(new HashMap<ActionObservation, Integer>());
	
	// reset the count of updating th, hist, C_ao
	protected void ResetCountsForNormalization()
	{
		CountNumberForC_ao.clear();
		CountNumberForHMats[0] = 0;
		CountNumberForTHMats[0] = 0;
	}
	
	public TPSR1(TrainingDataSet trainData, double minSingularVal, int maxDim, int maxHistLen, int maxTestLen)
	{
		super(trainData);
		
		if(maxHistLen > UPPER_HIST_LEN_LIMIT)
			throw new IllegalArgumentException("Max history length of: " + maxHistLen + 
					" exceeds limit of: " + UPPER_HIST_LEN_LIMIT);
		
		this.minSingularVal = minSingularVal;
		this.maxDim = maxDim;
		this.maxHistLen = maxHistLen;
		observers = new LinkedList<PSRObserver>();
		this.maxTestLen = maxTestLen;
	}
	
	public TPSR1(TrainingDataSet trainData, double minSingularVal, int maxDim)
	{
		super(trainData);
		this.minSingularVal = minSingularVal;
		this.maxDim = maxDim;
		this.maxHistLen = UPPER_HIST_LEN_LIMIT;
		observers = new LinkedList<PSRObserver>();
	}
	
	public void addPSRObserver(PSRObserver observer)
	{
		observers.add(observer);
	}
	
	protected void notifyPSRObservers(double[] singVals)
	{
		for(PSRObserver observer : observers)
		{
			observer.modelUpdated(singVals, tests, histories);
		}
	}
	
	protected Map<ActionObservation, DoubleMatrix> CaoMats;
	
	@Override
	protected void performBuild() throws Exception 
	{
		// Reset the counts for Σ_H Σ_TH C_ao
//		ResetCountsForNormalization();
		//initializing observable matrices, Σ_H Σ_TH
		if (th==null && hist == null)
		{
			hist = new DoubleMatrix(histories.size()+1, 1);
			th = new DoubleMatrix(tests.size(), histories.size()+1);
		}
		else
		{
			DoubleMatrix t = DoubleMatrix.zeros(tests.size(), histories.size()+1);
			DoubleMatrix h = DoubleMatrix.zeros(1, histories.size()+1);
			
			if (t.getRows()==th.getRows()&&t.getColumns()==th.getColumns())
			{}
			else
			{
				int[] indice = new int[th.getColumns()];
				for (int i = 0; i < th.getColumns(); i++)
				{
					indice[i] = i;
				}
				for (int row = 0; row < th.getRows(); row++)
				{
					t.put(row, indice, th.getRow(row));
				}
				th = t;
			}
			if (h.getColumns() == hist.getRows())
			{}
			else
			{
				int[] indice = new int[hist.getRows()];
				for (int i = 0; i < hist.getRows(); i++)
				{
					indice[i] = i;
				}
				h.put(0, indice, hist.transpose());
				hist = h.transpose();
			}
		}
		int num = Param.incrementalThreadsForPSR * (trainData.getBatchNumber()+1);
		while (1000%num!=0)
		{
			num--;
		}
		int threadsNum = Math.min(Param.MaximumThreadsForComputingPSR, num);
		// filling in hist and th matrix 
//		estimateHistoryAndTHMatricesMultiThread_FullyRebuild(threadsNum);
		// fixing the thread num because it only needs to iterate one batch of data
		estimateHistoryAndTHMatricesMultiThread(20);
		/*
		 *  normalize th, hist
		 *  CountNumberForHMats is the times of updating hist matrix
		 *  CountNumberForThMats is the times of updating th matrix
		 */
//		DoubleMatrix normhist = hist.mmul(1.0/(CountNumberForHMats[0]));
//		DoubleMatrix normth = th.mmul(1.0/(CountNumberForTHMats[0]));
		DoubleMatrix normhist = hist;
		DoubleMatrix normth = th;
		/*
		 * debug block for measuring likelihoods
		 */
//		for (int idx = 0; idx < histories.size(); idx++)
//		{
//			List<ActionObservation> historty = histories.getTestByCounter(idx);
//			if (historty.size()==1)
//			{
//				Map<List<ActionObservation>, Double> recordProbFornullhistory = new HashMap<List<ActionObservation>, Double>();
//				for (int i = 0; i < th.getRows(); i++)
//				{
//					double prob = th.get(i, idx + 1);
//					List<ActionObservation> t = tests.getTestByCounter(i);
//					recordProbFornullhistory.put(t, prob);
//				}
//				String FileName = historty.get(0).toString();
//				System.out.println(FileName + ":" + Double.toString(hist.get(idx + 1, 0)));
//				System.out.println(FileName + ":" + Double.toString(histBackup.get(idx + 1, 0)));
//				WriteRecordtoExcel(recordProbFornullhistory, FileName);
//			}
//		}
		System.out.println("Having Constructed TH Matrix!");
		
		// Taking SVD of th and initialize aoMats
		svdResults = computeTruncatedSVD(normth, minSingularVal, maxDim);
		pseudoInverse = Solve.pinv(svdResults[0].mmul(normth));
		aoMats = Collections.synchronizedMap(new HashMap<ActionObservation, DoubleMatrix>());
		CaoMats = new HashMap<ActionObservation, DoubleMatrix>();
		// debuging block for computing P_T,ao,H
		for(ActionObservation actOb : trainData.getValidActionObservationSet())
		{
			aoMats.put(actOb, new DoubleMatrix(pseudoInverse.getColumns(), pseudoInverse.getColumns()));
			CaoMats.put(actOb, DoubleMatrix.zeros(th.getRows(), th.getColumns()));
		}
		// Computing first part of equation 41

//		constructAOMatricesMuliThread(threadsNum);
		/*
		 * normalize C_ao
		 * ResetCount is the number of times that C_ao has been update. In other word, it is equal to the number of currentSequence = (any)hj + (special)ao + (any)ti
		 */
//		WriteExcel(CountNumberForC_ao);

		constructAOMatricesMuliThread_FullyRebuild(threadsNum);
		WriteDoubleMatrixToExcel(normth, "null history");
		for (ActionObservation actob:CaoMats.keySet()) {
			WriteDoubleMatrixToExcel(CaoMats.get(actob), actob.toString());
		}
		DoubleMatrix PQ = svdResults[0].mmul(normth).mmul(Solve.pinv(DoubleMatrix.diag(normhist)));
		DoubleMatrix minf = Solve.solveLeastSquares(PQ.transpose(), DoubleMatrix.ones(hist.getRows(), 1)).transpose();
		mInf = new Minf(minf.transpose());
		pv = PredictionVector.BuildPredctiveVector(PQ.getColumn(0));
		DoubleMatrix e = mInf.getVector().transpose().mmul(PQ);
		System.out.println(e);
//		WriteDoubleMatrixToExcel(CaoMats);
		
//		for (ActionObservation ao:aoMats.keySet())
//		{
//			DoubleMatrix C_ao = aoMats.get(ao);
//			DoubleMatrix P_ao = CaoMats.get(ao);
////			double ResetCount = (double) CountNumberForC_ao.get(ao);
////			C_ao.muli((double)(1.0/ResetCount));
////			P_ao.muli((double) (1.0/ResetCount));
//			if (!C_ao.equals(svdResults[0].mmul(P_ao).mmul(pseudoInverse)))
//			{
//				int a = 1;
//				a = 2;
//			}
//		}
	}
	
	protected void performUpdate()
	{
//		hist = DoubleMatrix.concatVertically(hist,  DoubleMatrix.zeros(histories.size()+1-hist.getRows(),1));
//		th = DoubleMatrix.zeros(th.getRows(), histories.size()+1);
		CountNumberForHMats[0] = 0;
		CountNumberForTHMats[0] = 0;
		hist = new DoubleMatrix(histories.size()+1, 1);
		th = DoubleMatrix.zeros(tests.size(), histories.size()+1);
	
//		estimateHistoryAndTHMatrices();
		estimateHistoryAndTHMatricesMultiThread(50);
//		double scaleHistory = (double)(1.0/CountNumberForHMats[0]);
//		double sumhist = hist.sum();
//		double scaleTH = (double)(1.0/CountNumberForTHMats[0]);
//		double sumTH = th.sum();
//		hist.mmuli(scaleHistory);
//		th.mmuli(scaleTH);
		DoubleMatrix oldU = svdResults[0].transpose().dup();
		DoubleMatrix oldS = svdResults[1].dup();
		DoubleMatrix oldV = svdResults[2].dup();
		svdResults = updateSVD(th);
//		svdResults = computeTruncatedSVD(th, minSingularVal, maxDim);
		oldV = DoubleMatrix.concatVertically(oldV, DoubleMatrix.zeros(svdResults[2].getRows()-oldV.getRows(), oldV.getColumns()));
		oldU = DoubleMatrix.concatVertically(oldU, DoubleMatrix.zeros(svdResults[0].getColumns()-oldU.getRows(), oldU.getColumns()));
		DoubleMatrix S_Inverse = Solve.pinv(svdResults[1]);
		pseudoInverse = S_Inverse.mmul(svdResults[2].transpose());
		for(ActionObservation actOb : trainData.getValidActionObservationSet())
		{
			if(!aoMats.keySet().contains(actOb))
			{
				aoMats.put(actOb, new DoubleMatrix(aoMats_rows, aoMats_columns));
			}
			else
			{
				DoubleMatrix mat = svdResults[0].mmul(oldU).mmul(aoMats.get(actOb)).mmul(oldS).mmul(oldV.transpose()).mmul(svdResults[2]).mmul(Solve.pinv(svdResults[1]));
				aoMats.put(actOb, mat);
			}
		}
		trainData.resetData();
		
//		constructAOMatrices();
		constructAOMatricesMuliThread(50);
		mInf = new Minf(((hist.transpose()).mmul(svdResults[2].mmul(S_Inverse))).transpose());
//		DoubleMatrix mat1 = svdResults[0].mmul(th).getColumn(0);
		DoubleMatrix mat2 = svdResults[1].mmul(svdResults[2].transpose()).getColumn(0);
//		if (!mat1.equals(mat2))
//		{
//			System.err.println("The mat1 is not equal to mat2");
//		}
		pv = PredictionVector.BuildPredctiveVector(mat2);
	}
	
	protected DoubleMatrix[] updateSVD(DoubleMatrix newTHData)
	{
		DoubleMatrix biggerV = DoubleMatrix.concatVertically(svdResults[2],
				DoubleMatrix.zeros(newTHData.getColumns()-svdResults[2].getRows(), svdResults[2].getColumns()));
//		DoubleMatrix zero = DoubleMatrix.zeros(svdResults[0].getRows(), newTHData.getRows()-svdResults[0].getColumns());
//		svdResults[0] = DoubleMatrix.concatHorizontally(svdResults[0], zero);
		DoubleMatrix m = svdResults[0].mmul(newTHData);
		DoubleMatrix p = newTHData.sub((svdResults[0].transpose()).mmul(m));
		
		DoubleMatrix pBase = Singular.sparseSVD(p)[0];
		pBase = DoubleMatrix.concatHorizontally(pBase, DoubleMatrix.zeros(pBase.getRows(), p.getColumns()-pBase.getColumns()));
		DoubleMatrix rA = (pBase.transpose()).mmul(p);
		
		DoubleMatrix n = (biggerV.transpose()).mmul(DoubleMatrix.eye(newTHData.getColumns()));
		DoubleMatrix q = DoubleMatrix.eye(newTHData.getColumns()).sub(biggerV.mmul(n));
		
		DoubleMatrix qBase = Singular.sparseSVD(q)[0];
		DoubleMatrix rB = (qBase.transpose()).mmul(q);
		
		DoubleMatrix z = DoubleMatrix.zeros(m.getRows(), m.getColumns());
		DoubleMatrix z2 = DoubleMatrix.zeros(m.getColumns(),m.getColumns());
		
		DoubleMatrix top = DoubleMatrix.concatHorizontally(svdResults[1],z);
		DoubleMatrix bottom = DoubleMatrix.concatHorizontally(z.transpose(),z2);
		DoubleMatrix comb = DoubleMatrix.concatVertically(top, bottom);
		
		DoubleMatrix mRA = DoubleMatrix.concatVertically(m, rA);
		DoubleMatrix nRB = DoubleMatrix.concatVertically(n, rB);
	
		DoubleMatrix mRANRB = mRA.mmul(nRB.transpose());
		DoubleMatrix k = comb.add(mRANRB);
		
		DoubleMatrix[] newBases = computeTruncatedSVD(k, Double.MIN_VALUE, maxDim);
				
		DoubleMatrix uP = (DoubleMatrix.concatHorizontally(svdResults[0].transpose(), p)).mmul(newBases[0].transpose());
		DoubleMatrix vP = (DoubleMatrix.concatHorizontally(biggerV,q)).mmul(newBases[2]);
		
		DoubleMatrix[] updatedSVDResults = new DoubleMatrix[3];
		
		updatedSVDResults[0] = uP.transpose();
		updatedSVDResults[1] = newBases[1];
		updatedSVDResults[2] = vP;
		
		double[] singVals = new double[updatedSVDResults[1].getRows()];
		for(int i = 0; i < updatedSVDResults[1].getRows(); i++)
		{
			singVals[i] = updatedSVDResults[1].get(i,i);
		}		
		return updatedSVDResults;
	}

	/*
	 * @Param MultiNum is the number of threads for updating C_ao
	 */
	protected void constructAOMatricesMuliThread_FullyRebuild(int MultiNum)
	{
		trainData.resetData();
		/*
		 * How many game experiences should be processed for a single thread
		 * Get number of games in one data batch, and distribute it into MultiNum Threads
		 */
		final int ResetCounterNum = trainData.getNumberOfRunsInBatch() / MultiNum;
		MultiThreader.foreach(new Range(MultiNum), new Functor<Integer, Map<ActionObservation, DoubleMatrix>>() {
			@Override
			public Map<ActionObservation, DoubleMatrix> function(Integer input) {
					// BatchNum starts from the maximum Batch and reduces to 0 for iterating all data instead of the latest Batch.
				try {
					int BatchNum = trainData.getBatchNumber();
					int RunCounter = ResetCounterNum * input;
					int stepCounter = 0;
					List<ActionObservation> currentSequence = new ArrayList<ActionObservation>();
					int count = 0;
					while(BatchNum>=0)
					{
						// if it has finished it's tasks
						while(count < ResetCounterNum)
						{
							// add one action observation pair into current sequence
							currentSequence.add(trainData.getNextActionObservationWithBatchNumrunCounterstepCounter(BatchNum, RunCounter, stepCounter));
							stepCounter ++;
							// if the game experience ends
							if (trainData.IsUpdateRunCounterAndstepCounter(BatchNum, RunCounter, stepCounter, 0))
							{
								stepCounter = 0;
								RunCounter = trainData.getUpdateRunCounter(BatchNum, RunCounter);
							}
							// update C_ao for null history and currentSequence == null + ao + ti
							setAONullHistories(currentSequence);
							// update C_ao for all possible combination of hj + ao + ti == currentSequence
							parseAndAddAOCounts(currentSequence);
							
							//checking if reset performed
							if(stepCounter == 0)
							{
								count++;
								currentSequence = new ArrayList<ActionObservation>();
							}
						}
						// Reset and starts to iterate previous data Batch
						BatchNum--;
						RunCounter =  ResetCounterNum * input;
						stepCounter = 0;
						count=0;
						currentSequence = new ArrayList<ActionObservation>();
				}
				} catch (Exception e) {
					e.printStackTrace();
				}
				return aoMats;
			}
		});
	}
	
	// updating the C_ao but only using lastest dataBatch
	protected void constructAOMatricesMuliThread(int MultiNum)
	{
		trainData.resetData();
		/*
		 * How many game experiences should be processed for a single thread
		 * Get number of games in one data batch, and distribute it into MultiNum Threads
		 */
		final int ResetCounterNum = trainData.getNumberOfRunsInBatch() / MultiNum;
		final int BatchNum = trainData.getBatchNumber();
		MultiThreader.foreach(new Range(MultiNum), new Functor<Integer, Map<ActionObservation, DoubleMatrix>>() {
			@Override
			public Map<ActionObservation, DoubleMatrix> function(Integer input) {
					// BatchNum starts from the maximum Batch and reduces to 0 for iterating all data instead of the latest Batch.
				try {
					int RunCounter = ResetCounterNum * input;
					int stepCounter = 0;
					List<ActionObservation> currentSequence = new ArrayList<ActionObservation>();
					int count = 0;
					// if it has finished it's tasks
					while(count < ResetCounterNum)
					{
						// add one action observation pair into current sequence
						currentSequence.add(trainData.getNextActionObservationWithBatchNumrunCounterstepCounter(BatchNum, RunCounter, stepCounter));
						stepCounter ++;
						// if the game experience ends
						if (trainData.IsUpdateRunCounterAndstepCounter(BatchNum, RunCounter, stepCounter, 0))
						{
							stepCounter = 0;
							RunCounter = trainData.getUpdateRunCounter(BatchNum, RunCounter);
						}
						// update C_ao for null history and currentSequence == null + ao + ti
						setAONullHistories(currentSequence);
						// update C_ao for all possible combination of hj + ao + ti == currentSequence
						parseAndAddAOCounts(currentSequence);
						//checking if reset performed
						if(stepCounter == 0)
						{
							count++;
							currentSequence = new ArrayList<ActionObservation>();
						}
					}
				} catch (Exception e) {
					e.printStackTrace();
				}
				return aoMats;
			}
		});
	}
	
	// update C_ao for all possible combination of hj + ao + ti == currentSequence
	protected void parseAndAddAOCounts(List<ActionObservation> currentSequence)
	{
		int hi;
		List<ActionObservation> currentHistory = new ArrayList<ActionObservation>();
		//looping through current sequence and parsing into possible sets of histories and tests
		for(int j = 0; j < currentSequence.size(); j++)
		{
			// if currentSequence[0, j+1] is out of limit, currentHistory = currentSequence[j + 1 - maxHistLen, j+1] 
			if (j < maxHistLen)
			{
				currentHistory = currentSequence.subList(0, j+1);
			}
			else
			{
				currentHistory = currentSequence.subList(j + 1 - maxHistLen, j+1);
			}
			hi = histories.indexOf(currentHistory)+1;
			
			if(j+2 < currentSequence.size())
			{
				ActionObservation actOb = currentSequence.get(j+1);
				List<ActionObservation> test;
				// make sure test's length is not out of limit
				if (currentSequence.size() > j + 2 + maxTestLen)
				{
					test =currentSequence.subList(j + 2, j + 2 + maxTestLen);
				}
				else
				{
					test =currentSequence.subList(j+2, currentSequence.size());
				}
				int ti = tests.indexOf(test);
				if (ti==-1)
				{
					if (!test.get(test.size()-1).equals(TrainingDataSet.getResetAO()))
					{
						System.err.print("The test:");
						for (int idx = 0; idx < test.size(); idx++)
						{
							System.err.print(test.get(idx));
						}
						System.err.println("doesn't have testID!");
					}
				}
				if (hi==0)
				{
					System.err.print("The history:");
					for (int idx = 0; idx < currentHistory.size(); idx++)
					{
						System.err.print(currentHistory.get(idx));
					}
					System.err.println("doesn't have historyID!");
				}
				// if test and history is not null, update C_ao by equation 38
				incrementAOMat(actOb, ti, hi);
			}
		}
	}
	
	// update C_ao when currentSequence = null + ao + ti
	protected void setAONullHistories(List<ActionObservation> currentSequence)
	{
		List<ActionObservation> seq = currentSequence.subList(1, currentSequence.size());
		ActionObservation actOb = currentSequence.get(0);
		int ti = tests.indexOf(seq);
		if (!seq.isEmpty()&&seq.size() <= maxTestLen && ti == -1)
		{
			if (!currentSequence.get(currentSequence.size()-1).equals(TrainingDataSet.getResetAO()))
			{
				System.err.print("The test:");
				for (int idx = 0; idx < seq.size(); idx++)
				{
					System.err.print(seq.get(idx));
				}
				System.err.println("doesn't have testID!");
			}
		}
		incrementAOMat(actOb, ti, 0);
	}
	
	protected boolean incrementAOMat(ActionObservation actOb, int testIndex, int histIndex)
	{
		if(testIndex == -1 || testIndex >= svdResults[0].getColumns())
		{
			if (testIndex >= svdResults[0].getColumns())
			{
				System.err.println("Not update aoMats, because testIndex:" + Integer.toString(testIndex));
			}
			return false;
		}
		synchronized (actOb) {
			DoubleMatrix aoMat = aoMats.get(actOb);
			DoubleMatrix colToAdd = svdResults[0].getColumn(testIndex);
			// Count the times of updates for each C_ao
			if (!CountNumberForC_ao.containsKey(actOb))
			{
				CountNumberForC_ao.put(actOb, 1);
			}
			else
			{
				CountNumberForC_ao.put(actOb, CountNumberForC_ao.get(actOb)+1);
			}
			// way1 to implement outer product
//			for(int rowIndex = 0; rowIndex < colToAdd.getRows(); rowIndex++)
//			{
//				double currEntry = colToAdd.get(rowIndex, 0);
//				DoubleMatrix currRowToAdd = pseudoInverse.getColumn(histIndex).dup();
//				currRowToAdd.muli(currEntry);
//				aoMat.putRow(rowIndex, aoMat.getRow(rowIndex).add(currRowToAdd));
//			}
			// way2 to implement outer product
			DoubleMatrix currRowToAdd = pseudoInverse.getRow(histIndex);
			DoubleMatrix tmp = colToAdd.mmul(currRowToAdd);
			aoMat.addi(tmp);
			if (CaoMats!=null)
			{
				DoubleMatrix CaoMat = CaoMats.get(actOb);
				CaoMat.put(testIndex, histIndex, CaoMat.get(testIndex, histIndex) + 1);
			}
			
		}
		return true;
	}
	
	protected DoubleMatrix[] computeTruncatedSVD(DoubleMatrix mat, Double singValTol, int maxSize)
	{
		DoubleMatrix[] svdResult = Singular.sparseSVD(mat);

		notifyPSRObservers(svdResult[1].toArray());
		
		int singIter;
		for(singIter = 0; singIter < svdResult[1].getRows(); singIter++)
		{
			double value = svdResult[1].get(singIter, 0);
			if(value < singValTol)
				break;
		}
		
		int numToKeep = Math.min(singIter, maxSize);
		if (numToKeep < maxSize)
		{
			System.err.println("The dimension is less than maxSize");
		}
		DoubleMatrix u = new DoubleMatrix(mat.getRows(), numToKeep);
		u = svdResult[0].getRange(0,svdResult[0].getRows(), 0, numToKeep);
		DoubleMatrix s = DoubleMatrix.diag(svdResult[1].getRange(0, numToKeep, 0, 1));
		DoubleMatrix v = new DoubleMatrix(mat.getColumns(), numToKeep);
		v = svdResult[2].getRange(0, svdResult[2].getRows(), 0, numToKeep);
		DoubleMatrix[] truncatedSVDResult = {u.transpose(),s,v};
		return truncatedSVDResult;
	}
	// Taking all batch data to fill in th and hist matrix
	protected void estimateHistoryAndTHMatricesMultiThread_FullyRebuild(int ThreaderNum)
	{
		trainData.resetData();
		/*
		 *  ResetCounterNum is how much game experiences should be processed for each single thread in one batch data. 
		 *  trainData.getNumberOfRunsInBatch() returns the number of games in one data batch
		 */
		final int ResetCounterNum = trainData.getNumberOfRunsInBatch() / ThreaderNum;
		List<List<DoubleMatrix>> MultiTHAoMatsList = MultiThreader.foreach(new Range(ThreaderNum), new Functor<Integer, List<DoubleMatrix>>() 
		{
					@Override
					public List<DoubleMatrix> function(Integer input) {
						List<DoubleMatrix> THMats = new ArrayList<DoubleMatrix>();
						try {
							// Starting from the lastest data batch and iterate the experiences to update th and hist matrix
							int BatchNum = trainData.getBatchNumber();
							int runCounter = ResetCounterNum * input;
							int stepCounter = 0;
							DoubleMatrix thThread = DoubleMatrix.zeros(th.getRows(), th.getColumns());
							DoubleMatrix histThread = DoubleMatrix.zeros(hist.getRows(), hist.getColumns());
							THMats.add(thThread);
							THMats.add(histThread);
							ArrayList<ActionObservation> currentSequence = new ArrayList<ActionObservation>();
							int count = 0;
							// if iterating all data batch
							while(BatchNum>=0)
							{
								// if finish the number of tasks in each data batch
								while (count < ResetCounterNum) {
									// add one action observation pair into current sequence
									currentSequence.add(trainData.getNextActionObservationWithBatchNumrunCounterstepCounter(BatchNum, runCounter, stepCounter));
									stepCounter++;
									// if the game ends
									if (trainData.IsUpdateRunCounterAndstepCounter(BatchNum, runCounter, stepCounter, 0))
									{
										stepCounter=0;
										runCounter = trainData.getUpdateRunCounter(BatchNum, runCounter);
									}
									// update hist matrix
//									histThread = incrementHistory(currentSequence, histThread);
									// update th matrix for currentSequence == null + ti
									setTHNullHistories(currentSequence, thThread, histThread);			
									// update th matrix of all combinations ti and hj for currentSequence == hj + ti 
									parseAndAddTHCounts(currentSequence, thThread, histThread);
									// checking if reset performed
									if (stepCounter == 0) {
										currentSequence = new ArrayList<ActionObservation>();
//										// adding one to prob of null history
//										synchronized (CountNumberForHMats) {
//											CountNumberForHMats[0]++;
//										}
//										if (Param.isRandomInit)
//										{
//											DoubleMatrix one = DoubleMatrix.ones(histThread.getRows(), histThread.getColumns());
//											histThread.putColumn(0, one.add(histThread.getColumn(0)));
//										}
//										else
//										{
//											histThread.put(0, 0, histThread.get(0, 0) + 1);
//										}
										count++;
									}
								}
								// reset and starts to iterate previous data batch
								BatchNum--;
								runCounter=ResetCounterNum * input;
								stepCounter=0;
								count=0;
								currentSequence = new ArrayList<ActionObservation>();
						}
						} catch (Exception e) {
							e.printStackTrace();
						}
						return THMats;
					}
				});
		
		// aggregate the outputs of different threads
		///////////////////////////////////////////////////////////
		for (int idx = 0; idx < MultiTHAoMatsList.size(); idx++)
		{
			DoubleMatrix th = MultiTHAoMatsList.get(idx).get(0);
			DoubleMatrix hist = MultiTHAoMatsList.get(idx).get(1);
			
			this.th.addi(th);
			this.hist.addi(hist);
		}
		///////////////////////////////////////////////////////////
	}
	
	protected void estimateHistoryAndTHMatricesMultiThread(int ThreaderNum)
	{
		trainData.resetData();
		/*
		 *  ResetCounterNum is how much game experiences should be processed for each single thread in one batch data. 
		 *  trainData.getNumberOfRunsInBatch() returns the number of games in one data batch
		 */
		final int BatchNum = trainData.getBatchNumber();
		final int ResetCounterNum = trainData.getNumberOfRunsInBatch() / ThreaderNum;
		List<List<DoubleMatrix>> MultiTHAoMatsList = MultiThreader.foreach(new Range(ThreaderNum), new Functor<Integer, List<DoubleMatrix>>() 
		{
					@Override
					public List<DoubleMatrix> function(Integer input) {
						List<DoubleMatrix> THMats = new ArrayList<DoubleMatrix>();
						try {
							int runCounter = ResetCounterNum * input;
							int stepCounter = 0;
							DoubleMatrix thThread = DoubleMatrix.zeros(th.getRows(), th.getColumns());
							DoubleMatrix histThread = DoubleMatrix.zeros(hist.getRows(), hist.getColumns());
							THMats.add(thThread);
							THMats.add(histThread);
							ArrayList<ActionObservation> currentSequence = new ArrayList<ActionObservation>();
							int count = 0;
								// if finish the number of tasks in each data batch
							while (count < ResetCounterNum) {
								// add one action observation pair into current sequence
								currentSequence.add(trainData.getNextActionObservationWithBatchNumrunCounterstepCounter(BatchNum, runCounter, stepCounter));
								stepCounter++;
								// if the game ends
								if (trainData.IsUpdateRunCounterAndstepCounter(BatchNum, runCounter, stepCounter, 0))
								{
									stepCounter=0;
									runCounter = trainData.getUpdateRunCounter(BatchNum, runCounter);
								}
								// update hist matrix
//									histThread = incrementHistory(currentSequence, histThread);
								// update th matrix for currentSequence == null + ti
								setTHNullHistories(currentSequence, thThread, histThread);			
								// update th matrix of all combinations ti and hj for currentSequence == hj + ti 
								parseAndAddTHCounts(currentSequence, thThread, histThread);
								// checking if reset performed
								if (stepCounter == 0) {
									currentSequence = new ArrayList<ActionObservation>();
									count++;
								}
							}
						} catch (Exception e) {
							e.printStackTrace();
						}
						return THMats;
					}
				});
		
		// aggregate the outputs of different threads
		///////////////////////////////////////////////////////////
		for (int idx = 0; idx < MultiTHAoMatsList.size(); idx++)
		{
			DoubleMatrix th = MultiTHAoMatsList.get(idx).get(0);
			DoubleMatrix hist = MultiTHAoMatsList.get(idx).get(1);
			
			this.th.addi(th);
			this.hist.addi(hist);
		}
		///////////////////////////////////////////////////////////
	}
	
	/**
	 * Helper method gets next action-observation pair
	 */
	protected void appendNextActionObservation(List<ActionObservation> currentSequence, TrainingDataSet trainData)
	{
		//getting next ActionObservation and adding to current sequence
		ActionObservation currentActionObservation = trainData.getNextActionObservation();
		currentSequence.add(currentActionObservation);
	}

	/**
	 * Helper method increments history count for this sequence
	 * 
	 * @param currentSequence The current sequence of action-observation pairs
	 */
	protected DoubleMatrix incrementHistory(List<ActionObservation> currentSequence, DoubleMatrix hist)
	{
		//incrementing history count for this sequence
		// if the history is over the limit of length, it is truncated as [end-MaxHistLen, end]
		int size = currentSequence.size();
		List<ActionObservation> constrainSequence;
		if (size > maxHistLen)
		{
			constrainSequence = currentSequence.subList(size - maxHistLen, size);
		}
		else
		{
			constrainSequence = currentSequence.subList(0, size);
		}
		int hi = histories.indexOf(constrainSequence)+1;
		// if history is not null
		if(hi != 0) 
		{
			synchronized (CountNumberForHMats) {
				CountNumberForHMats[0]++;
			}
			hist.put(hi, 0,  hist.get(hi, 0)+1);
		}
		else if (!currentSequence.get(currentSequence.size()-1).equals(TrainingDataSet.getResetAO()) && currentSequence.size() < maxHistLen)
		{
			System.err.print("The history:");
			for (int idx = 0; idx < currentSequence.size(); idx++)
			{
				System.err.print(currentSequence.get(idx));
			}
			System.err.println("doesn't have historyID!");
		}
		return hist;
	}

	/**
	 * Helper method sets the null history columns of observable matrices
	 * 
	 * @param currentSequence The current sequence of action-observation pairs.  
	 */
	protected DoubleMatrix setTHNullHistories(List<ActionObservation>  currentSequence, DoubleMatrix th)
	{		
			int ti = tests.indexOf(currentSequence);
			if (ti == -1 && currentSequence.size() <= maxTestLen)
			{
				ActionObservation lastao = currentSequence.get(currentSequence.size()-1);
				if (!lastao.equals(TrainingDataSet.getResetAO()))
				{
					System.err.print("The test:");
					for (int idx = 0; idx < currentSequence.size(); idx++)
					{
						System.err.print(currentSequence.get(idx));
					}
					System.err.println("doesn't have testID!");
				}
			}
			return addTHCount(ti, 0, th);
	}	
	/*
	 * debugging block
	 */
	protected void setTHNullHistories(List<ActionObservation>  currentSequence, DoubleMatrix th, DoubleMatrix hist)
	{		
			int ti = tests.indexOf(currentSequence);
			if (ti == -1 && currentSequence.size() <= maxTestLen)
			{
				ActionObservation lastao = currentSequence.get(currentSequence.size()-1);
				if (!lastao.equals(TrainingDataSet.getResetAO()))
				{
					System.err.print("The test:");
					for (int idx = 0; idx < currentSequence.size(); idx++)
					{
						System.err.print(currentSequence.get(idx));
					}
					System.err.println("doesn't have testID!");
				}
			}
			addTHCount(ti, 0, th, hist);
	}
	
	/**
	 * Helper method parses sequence into tests and histories and add counts
	 * 
	 * @param currentSequence The current sequence of action-observation pairs
	 */
	protected DoubleMatrix parseAndAddTHCounts(List<ActionObservation> currentSequence, DoubleMatrix th)
	{
		int hi=0;
		List<ActionObservation> currentHistory = new ArrayList<ActionObservation>();

		//looping through current sequence and parsing into possible sets of histories and tests
		for(int j = 0; j < currentSequence.size(); j++)
		{
			// history is not over the limit of length, otherwise, truncate it
			if (j < maxHistLen)
			{
				currentHistory = currentSequence.subList(0, j+1);
			}
			else
			{
				currentHistory = currentSequence.subList(j + 1 - maxHistLen, j+1);
			}
			hi = histories.indexOf(currentHistory)+1;
			if(j+1 < currentSequence.size())
			{
				List<ActionObservation> test;
				// test is not over the limit of length, otherwise, truncate it
				if (currentSequence.size() > j + 1 + maxTestLen) 
				{
					test = currentSequence.subList(j + 1, j + 1 + maxTestLen);
				}
				else 
				{
					test = currentSequence.subList(j + 1, currentSequence.size());
				}
				int ti=-1;
				ti = tests.indexOf(test);
	
				if (ti == -1)
				{
					if (!test.get(test.size()-1).equals(TrainingDataSet.getResetAO()))
					{
						System.err.print("The test:");
						for (int idx = 0; idx < test.size(); idx++)
						{
							System.err.print(test.get(idx));
						}
						System.err.println("doesn't have testID!");
					}
				}
				if (hi == 0)
				{
					System.err.print("The history:");
					for (int idx = 0; idx < currentHistory.size(); idx++)
					{
						System.err.print(currentHistory.get(idx));
					}
					System.err.println("doesn't have historyID!");
				}
				// update th matrix
				th = addTHCount(ti, hi, th);
			}
		}
		return th;
	}
	
	protected void parseAndAddTHCounts(List<ActionObservation> currentSequence, DoubleMatrix th, DoubleMatrix hist)
	{
		int hi=0;
		List<ActionObservation> currentHistory = new ArrayList<ActionObservation>();

		//looping through current sequence and parsing into possible sets of histories and tests
		for(int j = 0; j < currentSequence.size(); j++)
		{
			// history is not over the limit of length, otherwise, truncate it
			if (j < maxHistLen)
			{
				currentHistory = currentSequence.subList(0, j+1);
			}
			else
			{
				currentHistory = currentSequence.subList(j + 1 - maxHistLen, j+1);
			}

			hi = histories.indexOf(currentHistory)+1;
			if(j+1 < currentSequence.size())
			{
				List<ActionObservation> test;
				// test is not over the limit of length, otherwise, truncate it
				if (currentSequence.size() > j + 1 + maxTestLen) 
				{
					test = currentSequence.subList(j + 1, j + 1 + maxTestLen);
				}
				else 
				{
					test = currentSequence.subList(j + 1, currentSequence.size());
				}
				int ti=-1;
				ti = tests.indexOf(test);
	
				if (ti == -1)
				{
					if (!test.get(test.size()-1).equals(TrainingDataSet.getResetAO()))
					{
						System.err.print("The test:");
						for (int idx = 0; idx < test.size(); idx++)
						{
							System.err.print(test.get(idx));
						}
						System.err.println("doesn't have testID!");
					}
				}
				if (hi == 0)
				{
					System.err.print("The history:");
					for (int idx = 0; idx < currentHistory.size(); idx++)
					{
						System.err.print(currentHistory.get(idx));
					}
					System.err.println("doesn't have historyID!");
				}
				// update th matrix
				addTHCount(ti, hi, th, hist);
			}
		}
	}

	/**
	 * Adds a th count
	 * 
	 * @param ti test index
	 * @param hi history index
	 */
	protected DoubleMatrix addTHCount(int ti, int hi, DoubleMatrix th)
	{
		if (ti > th.getRows() || hi > th.getColumns())
		{
			System.err.println("ti or hi are greater than th");
		}
		if(ti != -1 && ti < th.getRows() && hi < th.getColumns())
		{
			synchronized (CountNumberForTHMats) {
				CountNumberForTHMats[0]++;
			}
			th.put(ti,hi, th.get(ti,hi)+1);
		}
		return th;
	}
	
	protected void addTHCount(int ti, int hi, DoubleMatrix th, DoubleMatrix hist)
	{
		if (ti > th.getRows() || hi > th.getColumns())
		{
			System.err.println("ti or hi are greater than th");
		}
		if(ti != -1 && ti < th.getRows() && hi < th.getColumns())
		{
			synchronized (CountNumberForTHMats) {
				CountNumberForTHMats[0]++;	
				CountNumberForHMats[0]++;
			}
			th.put(ti,hi, th.get(ti, hi)+1);
			hist.put(hi, 0, hist.get(hi, 0) + 1);
		}
	}

	@Override
	public void loadingExistPSR(String pSRPath) throws IOException, ParseException {
		// TODO Auto-generated method stub
		
	}
	
	/**
	 * Helper method gets next action-observation pair
	 */
	protected void appendNextActionObservation(List<ActionObservation> currentSequence, int batch)
	{
		//getting next ActionObservation and adding to current sequence
		ActionObservation currentActionObservation = trainData.getNextActionObservation(batch);
		currentSequence.add(currentActionObservation);
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	@Override
	public void update_traindata(TrainingDataSet trainData2) {
		// TODO Auto-generated method stub
		this.trainData = trainData2;
	}
	private void WriteRecordtoExcel(Map<List<ActionObservation>, Double> LikelihoodsTestsForNullHistory, String filename)
	{
		HSSFWorkbook workbook = new HSSFWorkbook();
        HSSFSheet sheet = workbook.createSheet(filename);
        Set<List<ActionObservation>> keyset = LikelihoodsTestsForNullHistory.keySet();
        int rownum=0;
		for (List<ActionObservation> key : keyset) {
            	Row row = sheet.createRow(rownum++);
	            double objArr = LikelihoodsTestsForNullHistory.get(key);
	            IntSeq ID = ActObSequenceSet.computeID(key);
	            int cellnum = 0;
	            Cell cell = row.createCell(cellnum++);
	            Cell cell1 = row.createCell(cellnum++);
	            Cell cell2 = row.createCell(cellnum++);
	            cell.setCellValue(key.toString());
	            cell1.setCellValue(objArr);
	            cell2.setCellValue(ID.toString());
		}
					  
        try {
            FileOutputStream out
                    = new FileOutputStream(new File(filename + ".xls"));
            workbook.write(out);
            out.close();
            System.out.println("Excel written successfully..");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
	}
	// debug functions, output the times of updating each ao
	
	private void WriteExcel(Map<ActionObservation, Integer> C_ao)
	{
		HSSFWorkbook workbook = new HSSFWorkbook();
        HSSFSheet sheet = workbook.createSheet("LikelihoodsAO");
        Map<ActionObservation, Integer> data = new LinkedHashMap<ActionObservation, Integer>(C_ao);   
        Set<ActionObservation> keyset = data.keySet();
        int rownum = 0;
		int aid = 0;
		while(aid < Param.actionSize)
		{
			int oid = 0;
			while (oid < Param.observationSize)
			{
				int rid = 0;
				while (rid < Param.rewardSize)
				{
					for (ActionObservation key : keyset) {
			            if (key.getAction().getID()==aid&&key.getObservation().getoID()==oid&&key.getObservation().getrID()==rid)
			            {
			            	Row row = sheet.createRow(rownum++);
				            Integer objArr = data.get(key);
				            int cellnum = 0;
				            Cell cell = row.createCell(cellnum++);
				            Cell cell1 = row.createCell(cellnum++);
				            cell.setCellValue(key.toString());
				            cell1.setCellValue(objArr);
				            break;
			            }
			        }
					rid++;
				}
				oid++;
			}
			aid++;
		}
        try {
            FileOutputStream out
                    = new FileOutputStream(new File("LikelihoodsAO.xls"));
            workbook.write(out);
            out.close();
            System.out.println("Excel written successfully..");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
	}
	String ConvertListStringToOne(List<ActionObservation> listactob)
	{
		String tmp = "";
		for (int i = 0; i < listactob.size(); i++)
		{
			tmp += listactob.get(i);
		}
		return tmp;
	}
	protected void WriteDoubleMatrixToExcel(DoubleMatrix aoMats, String actob) throws IOException
	{
		FileWriter csvReader = new FileWriter(actob + ".csv");
		csvReader.append("Test");
		csvReader.append(",");
		csvReader.append("null history");
		csvReader.append(",");

		for (int hid = 1; hid < histories.size() + 1; hid++)
		{
			csvReader.append(ConvertListStringToOne(histories.getTestByCounter(hid - 1)));
			if (hid == histories.size())
			{
				csvReader.append("\n");
			}
			else
			{
				csvReader.append(",");
			}
		}
		for (int rowid = 0; rowid < aoMats.getRows(); rowid++)
		{
			csvReader.append(ConvertListStringToOne(tests.getTestByCounter(rowid)));
			csvReader.append(",");
			for (int colid = 0; colid < aoMats.getColumns(); colid++)
			{
				csvReader.append(Double.toString(aoMats.get(rowid, colid)));
				if (colid == aoMats.getColumns() - 1)
				{
					csvReader.append("\n");
				}
				else
				{
					csvReader.append(",");
				}
			}
		}
		csvReader.close();
	}
	
//	protected void WriteDoubleMatrixToExcel(Map<ActionObservation, DoubleMatrix> C_ao)
//	{
//		Workbook workbook = new XSSFWorkbook();
//		for (ActionObservation ao : C_ao.keySet())
//		{
//			String filename = "Cao" + ao.toString();
//			DoubleMatrix aoMat = C_ao.get(ao);
//	        Sheet sheet = workbook.createSheet(filename);
//	        Row historyname = sheet.createRow(0);	
//	        for (int idx = 0; idx < histories.size(); idx++)
//	        {
//	        	XSSFCell cell = historyname.createCell(idx + 2);
//	        	try {
//	        	cell.setCellValue(histories.getTestByCounter(idx).toString());
//	        	}catch(Exception e)
//	        	{
//	        		e.printStackTrace();
//	        	}
//	        }
//	        for (int idx = 0; idx < tests.size(); idx++)
//	        {
//	        	Row testName = sheet.createRow(idx + 1);
//	        	Cell cell = testName.createCell(0);
//	        	String str = tests.getTestByCounter(idx).toString();
//	        	cell.setCellValue(str);
//	        }
//	        for (int rownum = 1; rownum < aoMat.getRows() + 1; rownum++)
//	        {
//	        	Row row = sheet.createRow(rownum);
//	        	for (int colnum = 1; colnum < aoMat.getColumns() + 1; colnum++)
//	        	{
//	            	Cell cell = row.createCell(colnum);
//	                cell.setCellValue(aoMat.get(rownum-1, colnum-1));
//	        	}
//	        }
//		}
//        try {
//            FileOutputStream out
//                    = new FileOutputStream(new File("Cao" + ".xls"));
//            workbook.write(out);
//            out.close();
//            System.out.println("Excel written successfully..");
//        } catch (FileNotFoundException e) {
//            e.printStackTrace();
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//	}
}