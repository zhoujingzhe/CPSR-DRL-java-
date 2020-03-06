package cpsr.model;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;
import Parameter.Param;
import cpsr.environment.TrainingDataSet;
import cpsr.environment.components.Action;
import cpsr.environment.components.ActionObservation;
import cpsr.model.CPSR.ProjType;
import cpsr.model.components.Minf;
import cpsr.model.components.PredictionVector;

public class MemEffCPSR_NotCompressingHistory extends TPSR 
{

	private DoubleMatrix one = null;
	private static final long serialVersionUID = -8582382328551442043L;

	protected static long SEED = 10;

	public static final int MAX_CACHE_SIZE = 100000;

	Map<Integer, DoubleMatrix>  colToAddCache, rowToAddCache;
	static Map<Integer, DoubleMatrix> randVecCache = Collections.synchronizedMap(new HashMap<Integer, DoubleMatrix>(10000, 10000));
	protected boolean randStart;

	protected ProjType projType;

	protected int projDim;
	DoubleMatrix P_th;
	public void update_traindata(TrainingDataSet TrainData)
	{
		this.trainData = TrainData;
	}

	public MemEffCPSR_NotCompressingHistory(TrainingDataSet trainData, double minSingularVal, int maxSVDDim, int projDim, ProjType projType, boolean randStart)
	{

		this(trainData, minSingularVal, maxSVDDim, projDim, CPSR.DEF_MAX_HIST, projType,  randStart, CPSR.DEF_MAX_Test);
	}

	public MemEffCPSR_NotCompressingHistory(TrainingDataSet trainData, double minSingularVal, int maxSVDDim, int projDim, int maxHistLen, ProjType projType, boolean randStart, int maxTestLen)
	{
		super(trainData, minSingularVal, maxSVDDim, maxHistLen, maxTestLen);
		this.projType = projType;
		this.projDim = projDim;
		one = DoubleMatrix.ones(projDim+1, 1);
	}

	/*
	 * Building the CPSR 
	 */
	@Override
	protected void performBuild() throws Exception 
	{
		// Reset the counts for Σ_H Σ_TH C_ao
//		ResetCountsForNormalization();
		// creating Cache for storing  UˆT φ(ti),  Sˆ{−1} VˆT φ(hj) for each ti and hj
		this.colToAddCache = Collections.synchronizedMap(new MaxSizeHashMap<Integer, DoubleMatrix>(100000, 10000));
		this.rowToAddCache = Collections.synchronizedMap(new MaxSizeHashMap<Integer, DoubleMatrix>(100000, 10000));
		P_th = DoubleMatrix.zeros(tests.size(), histories.size() + 1);
		//initializing observable matrices, Σ_H Σ_TH
		if (hist == null && th == null)
		{
			th = new DoubleMatrix(projDim, histories.size() + 1);
			hist = new DoubleMatrix(histories.size() + 1, 1);
		}
		else
		{
			DoubleMatrix t = DoubleMatrix.zeros(projDim, histories.size() + 1);
			DoubleMatrix h = DoubleMatrix.zeros(1, histories.size() + 1);
			
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
		// fixing the length is because it only needs to iterate one batch of data
		estimateHistoryAndTHMatricesMultiThread(10);
//		estimateHistoryAndTHMatricesMultiThread_FullyRebuild(threadsNum);
		/*
		 *  normalize th, hist
		 *  CountNumberForHMats is the times of updating hist matrix
		 *  CountNumberForThMats is the times of updating th matrix
		 */
//		hist.mmuli(1.0/(CountNumberForHMats[0]));
//		th.mmuli(1.0/(CountNumberForTHMats[0]));
		System.out.println("Having Constructed TH Matrix!");
		
		// Taking SVD of th and initialize aoMats
		svdResults = computeTruncatedSVD(th, minSingularVal, maxDim);
		pseudoInverse = Solve.pinv(svdResults[0].mmul(th));
		aoMats = Collections.synchronizedMap(new HashMap<ActionObservation, DoubleMatrix>());
		CaoMats = new HashMap<ActionObservation, DoubleMatrix>();
		for(ActionObservation actOb : trainData.getValidActionObservationSet())
		{
			aoMats.put(actOb, new DoubleMatrix(pseudoInverse.getColumns(), pseudoInverse.getColumns()));
			CaoMats.put(actOb, DoubleMatrix.zeros(tests.size(), histories.size() + 1));
		}
		
		// Computing first part of equation 41
		constructAOMatricesMuliThread_FullyRebuild(threadsNum);
//		WriteDoubleMatrixToExcel(P_th, "null history");
//		for (ActionObservation actob:CaoMats.keySet())
//		{
//			WriteDoubleMatrixToExcel(CaoMats.get(actob), actob.toString());
//		}
		
		/*
		 * normalize C_ao
		 * ResetCount is the number of times that C_ao has been update. In other word, it is equal to the number of currentSequence = (any)hj + (special)ao + (any)ti
		 */
//		WriteExcel(CountNumberForC_ao);

//		for (ActionObservation ao:aoMats.keySet())
//		{
//			DoubleMatrix C_ao = aoMats.get(ao);
//			int ResetCount = CountNumberForC_ao.get(ao);
//			C_ao.muli((double)(1.0/ResetCount));
//			aoMats.put(ao, C_ao);
//		}
		//mInf = ΣˆT_H V Sˆ{−1}, seeing equation 37
		DoubleMatrix PQ = svdResults[0].mmul(th).mmul(Solve.pinv(DoubleMatrix.diag(hist)));
		DoubleMatrix minf = Solve.solveLeastSquares(PQ.transpose(), DoubleMatrix.ones(hist.getRows(), 1));
		mInf = new Minf(minf);
		pv = PredictionVector.BuildPredctiveVector(PQ.getColumn(0));
//		DoubleMatrix e = mInf.getVector().transpose().mmul(PQ);
//		System.out.println(e);
	}

	/*
	 * Incremental update the CPSR model
	 */
	protected void performUpdate()
	{
		// Renew Cache for storing  UˆT φ(ti),  Sˆ{−1} VˆT φ(hj) for each ti and hj
		this.colToAddCache = Collections.synchronizedMap(new MaxSizeHashMap<Integer, DoubleMatrix>(100000, 10000));
		this.rowToAddCache = Collections.synchronizedMap(new MaxSizeHashMap<Integer, DoubleMatrix>(100000, 10000));
		int threadsNum = Param.incrementalThreadsForPSR;
		// Reset the counts for Σ_H Σ_TH C_ao
		ResetCountsForNormalization();
		
		// Renew th matrix
		th = new DoubleMatrix(th.getRows(), th.getColumns());
		// Taking all trajectories to fill in th Matrix
		estimateHistoryAndTHMatricesMultiThread(threadsNum);
		System.out.println("Having Constructed TH Matrix!");
		
		//  C_ao=UˆT_new U_old Cao_{old}  S_old VˆT_old V_new Sˆ{−1}_new, which is second part of equation 41
		DoubleMatrix oldU = svdResults[0].transpose().dup();
		DoubleMatrix oldS = svdResults[1].dup();
		DoubleMatrix oldV = svdResults[2].dup();
		svdResults = updateSVD(th);

		oldV = DoubleMatrix.concatVertically(oldV, DoubleMatrix.zeros(svdResults[2].getRows()-oldV.getRows(), oldV.getColumns()));
		DoubleMatrix S_Inverse = Solve.pinv(svdResults[1]);
		pseudoInverse = S_Inverse.mmul(svdResults[2].transpose());
		for(ActionObservation actOb : trainData.getValidActionObservationSet())
		{
			if(!aoMats.keySet().contains(actOb))
			{
				aoMats.put(actOb, new DoubleMatrix(pseudoInverse.getColumns(), pseudoInverse.getColumns()));
			}
			else
			{
				DoubleMatrix old_aoMats = aoMats.get(actOb);
				if (old_aoMats.min() == 0.0 || old_aoMats.max() == 0.0) {
					System.err.println("The aoMats is zeros, actOb:" + actOb);
				}
				DoubleMatrix aoMat = svdResults[0].mmul(oldU).mmul(aoMats.get(actOb)).mmul(oldS).mmul(oldV.transpose()).mmul(svdResults[2]).mmul( Solve.pinv(svdResults[1]));
				if (aoMat.min() == 0.0 || aoMat.max() == 0.0) {
					System.err.println("The base of aoMats is zeros, checking performupdate function!");
				}
				aoMats.put(actOb, aoMat);
			}
		}
		
//		constructAOMatrices();
		// Computing first part of equation 41
		constructAOMatricesMuliThread(threadsNum);
		
		//mInf = ΣˆT_H V Sˆ{−1}, seeing equation 37
		mInf = new Minf(((hist.transpose()).mmul(svdResults[2].mmul(S_Inverse))).transpose());
		if (Param.isRandomInit)
		{
			// if agent randomly starts, pv = S VˆT e, e=[1, 1, 1, 1, 1]
			pv = PredictionVector.BuildPredctiveVector(svdResults[1].mmul(svdResults[2].transpose().mmul(one)));
		}
		else
		{
			// if agent starts from a special position, e = [1, 0, 0, 0, 0]
			pv = PredictionVector.BuildPredctiveVector(svdResults[1].mmul(svdResults[2].transpose()).getColumn(0));
		}
	}

	/*
	 * Incremental update the value on C_{ao}, see equation-38
	 * colToAddCache contains the UˆT φ(ti) for each ti
	 * rowToAddCache contains the Sˆ{−1} VˆT φ(hj) for each hj
	 */
//	protected boolean incrementAOMat(ActionObservation actOb, int testIndex, int histIndex)
//	{
//		// if testIndex == -1, it means the test is null or out of the limit(length over the constraint) 
//		if(testIndex == -1)
//			return false;
//		
//		DoubleMatrix aoCoeffecientsMat;
//		// synchronized locker, in case two threads simultaneously modify a same C_ao
//		synchronized (actOb) {
//			// Count the times of updates for each C_ao
//			if (!CountNumberForC_ao.containsKey(actOb))
//			{
//				CountNumberForC_ao.put(actOb, 1);
//			}
//			else
//			{
//				CountNumberForC_ao.put(actOb, CountNumberForC_ao.get(actOb)+1);
//			}
//			aoCoeffecientsMat = aoMats.get(actOb);
//			
//			DoubleMatrix colToAdd;
//			if (colToAddCache.containsKey(testIndex)) 
//			{
//				colToAdd = colToAddCache.get(testIndex);
//			} 
//			else
//			{
//				colToAdd = svdResults[0].mmul(getRandomVector(projDim, projType, testIndex));
//				colToAddCache.put(testIndex, colToAdd);		
//			}
//
//			DoubleMatrix currRowToAdd;
//			if (rowToAddCache.containsKey(histIndex)) 
//			{
//				currRowToAdd = rowToAddCache.get(histIndex);
//			}
//			else 
//			{
//				DoubleMatrix randHistVector;
////				if (histIndex == 0) 
////				{
////					// if the history is null, when the agent randomly starts, the φ(hj) = [1,1,1,...].
////					if (Param.isRandomInit)
////					{
////						randHistVector = DoubleMatrix.ones(projDim + 1, 1);
////					}
////					else
////					{
////						// When the agent starts from a special position, the φ(hj) = [1, 0, 0, 0...]
////						randHistVector = DoubleMatrix.zeros(projDim + 1, 1);
////						randHistVector.put(0, 0, 1);
////					}
////				} 
////				else
////				{
////					// if the history is not null, generate a random vector and expand an extra zero.
////					randHistVector = getRandomVector(projDim, projType, -1 * histIndex);
////					randHistVector = DoubleMatrix.concatVertically(DoubleMatrix.zeros(1), randHistVector);
////				}
//				// Computing Sˆ{−1} VˆT φ(hj)
////				currRowToAdd = pseudoInverse.mmul(randHistVector);
////				synchronized (rowToAddCache) {
////					if (rowToAddCache.containsKey(histIndex))
////					{
////						currRowToAdd = rowToAddCache.get(histIndex);
////					}
////					else
////					{
//						currRowToAdd = pseudoInverse.getRow(histIndex);
//						rowToAddCache.put(histIndex, currRowToAdd);
////					}
//				}
//			}
//			// Taking outer product
////			for (int rowIndex = 0; rowIndex < colToAdd.getRows(); rowIndex++)
////			{
////				double currEntry = colToAdd.get(rowIndex, 0);
////				DoubleMatrix tmp = currRowToAdd.mmul(currEntry);
////				DoubleMatrix updated_row = aoCoeffecientsMat.getRow(rowIndex);
////				updated_row = updated_row.add(tmp.transpose());
////				aoCoeffecientsMat.putRow(rowIndex, updated_row);
////			}
//			// way 2 implement outer
//			DoubleMatrix tmp = colToAdd.mmul(currRowToAdd);
//			aoCoeffecientsMat.addi(tmp);
//			aoMats.put(actOb, aoCoeffecientsMat);
//		}
//		
//		// Checking the C_ao is not all zero
//		if (aoCoeffecientsMat.min() == 0.0 || aoCoeffecientsMat.max() == 0.0) {
//			System.err.println("Error on IncrementAOMats!");
//		}
//		return true;
//	}
	protected boolean incrementAOMat(ActionObservation actOb, int testIndex, int histIndex)
	{
		// if testIndex == -1, it means the test is null or out of the limit(length over the constraint) 
		if(testIndex == -1)
			return false;
		
		DoubleMatrix aoCoeffecientsMat;
		// synchronized locker, in case two threads simultaneously modify a same C_ao
		synchronized (actOb) {
			// Count the times of updates for each C_ao
			if (!CountNumberForC_ao.containsKey(actOb))
			{
				CountNumberForC_ao.put(actOb, 1);
			}
			else
			{
				CountNumberForC_ao.put(actOb, CountNumberForC_ao.get(actOb)+1);
			}
			aoCoeffecientsMat = aoMats.get(actOb);
			
			DoubleMatrix colToAdd;
			if (colToAddCache.containsKey(testIndex)) 
			{
				colToAdd = colToAddCache.get(testIndex);
			} 
			else
			{
				colToAdd = svdResults[0].mmul(getRandomVector(projDim, projType, testIndex));
				colToAddCache.put(testIndex, colToAdd);		
			}

			DoubleMatrix currRowToAdd;
			if (rowToAddCache.containsKey(histIndex)) 
			{
				currRowToAdd = rowToAddCache.get(histIndex);
			}
			else 
			{
				currRowToAdd = pseudoInverse.getRow(histIndex);
				rowToAddCache.put(histIndex, currRowToAdd);
			}
			DoubleMatrix tmp = colToAdd.mmul(currRowToAdd);
			aoCoeffecientsMat.addi(tmp);
			aoMats.put(actOb, aoCoeffecientsMat);
			CaoMats.get(actOb).put(testIndex, histIndex, CaoMats.get(actOb).get(testIndex, histIndex) + 1);
		}
		
		// Checking the C_ao is not all zero
		if (aoCoeffecientsMat.min() == 0.0 || aoCoeffecientsMat.max() == 0.0) {
			System.err.println("Error on IncrementAOMats!");
		}
		return true;
	}

	/**
	 * Adds a th count
	 * @param ti test index
	 * @param hi history index
	 * Updating ΣˆT ,H
	 */
	protected void addTHCount(int ti, int hi, DoubleMatrix th, DoubleMatrix hist)
	{
		// if ti is -1 means ti is null or out of the constraint
		if(ti != -1)
		{
			// Count the times of updating the th matrix
			synchronized (CountNumberForTHMats) {
				CountNumberForTHMats[0]++;
			}
			
			//  colToAdd is a random vector φ(ti) for ti
			DoubleMatrix colToAdd = null;
			colToAdd = getRandomVector(projDim, projType, ti);
			if (colToAdd == null)
			{
				System.err.println("colToAdd is null");
			}	
			// currRowToAdd is a random vector φ(hi) for hi
			th.putColumn(hi, th.getColumn(hi).add(colToAdd));
			synchronized (P_th) {
				P_th.put(ti, hi, P_th.get(ti, hi) + 1);
			}
			incrementHistory(hi, hist);
		}
	}

	/**
	 * Helper method increments history count for this sequence
	 * updating ΣˆH
	 * @param currentSequence The current sequence of action-observation pairs
	 */
	protected void incrementHistory(int hi, DoubleMatrix hist)
	{
		// if history is not null, update ΣˆH matrix
			// Count the times of updating the hist matrix
		synchronized (CountNumberForHMats)
		{
			CountNumberForHMats[0]++;
		}
		hist.put(hi, 0, hist.get(hi, 0) + 1);
	}
	/*
	 *  generate a random vector for each test and history
	 *  Check does it need to divide sqrt(projDim) https://www.stat.berkeley.edu/~mmahoney/f13-stat260-cs294/Lectures/lecture05.pdf page 7.
	 */
	@SuppressWarnings("incomplete-switch")
	protected DoubleMatrix getRandomVector(int projDim, ProjType projType, int index) {
		if (randVecCache.containsKey(index)) {
			return randVecCache.get(index);
		} else {
			synchronized (randVecCache) {
				if (randVecCache.containsKey(index)) {
					return randVecCache.get(index);
				}
				DoubleMatrix randVec = new DoubleMatrix(projDim, 1);
//				Random rand = new Random(index * 1);
				Random rand = new Random(Param.getRandomSeed());
				int rows = randVec.getRows();
				switch (projType) {
				case Spherical:
					// setting entries to random numbers drawn from gaussian distrubution.
					for (int i = 0; i < rows; i++) {
						randVec.put(i, 0, rand.nextGaussian() / Math.sqrt(((double) projDim)));
//						randVec.put(i, 0, rand.nextGaussian() / ((double) projDim));
					}
					break;
				case Bernoulli:
					// setting entries to random numbers drawn from gaussian distrubution.
					for (int i = 0; i < rows; i++) {
						if (rand.nextBoolean()) {
							randVec.put(i, 0, 1.0 / ((double) projDim));
						} else {
							randVec.put(i, 0, -1.0 / ((double) projDim));
						}
					}
					break;
				case ModifiedBernoulli:
					// setting entries to random numbers drawn from gaussian distrubution.
					for (int i = 0; i < rows; i++) {
						Double randDub = rand.nextDouble();
						if (randDub < 1.0 / 6.0) {
							randVec.put(i, 0, 1.0);
						} else if (randDub > 5.0 / 6.0) {
							randVec.put(i, 0, -1.0);
						} else {
							randVec.put(i, 0, 0);
						}
					}
				}
				randVecCache.put(index, randVec);
				return randVecCache.get(index);
			}
		}
	}
	
	public static class MaxSizeHashMap<K, V> extends LinkedHashMap<K, V> {
		private static final long serialVersionUID = 6701902928132006331L;
		private final int maxSize;

		public MaxSizeHashMap(int maxSize, int initCapacity) {
			super(initCapacity, 0.75f, true);
			this.maxSize = maxSize;
		}

		@Override
		protected boolean removeEldestEntry(Map.Entry<K, V> eldest) {
			return size() > maxSize;
		}
	}
}


