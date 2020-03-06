package cpsr.model;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;
import Parameter.Param;
import cpsr.environment.TrainingDataSet;
import cpsr.environment.components.ActionObservation;
import cpsr.model.CPSR.ProjType;
import cpsr.model.components.Minf;
import cpsr.model.components.PredictionVector;

public class MemEffCPSR extends TPSR 
{

	private DoubleMatrix one = null;
	private static final long serialVersionUID = -8582382328551442043L;

	protected static long SEED = 10;

	public static final int MAX_CACHE_SIZE = 100000;

	Map<Integer, DoubleMatrix>  colToAddCache, rowToAddCache;
	Map<ActionObservation, DoubleMatrix> PaoMats;
	static Map<Integer, DoubleMatrix> randVecCache = Collections.synchronizedMap(new HashMap<Integer, DoubleMatrix>(10000, 10000));
	protected boolean randStart;
	private DoubleMatrix e;
	protected ProjType projType;

	protected int projDim;

	public void update_traindata(TrainingDataSet TrainData)
	{
		this.trainData = TrainData;
	}

	public MemEffCPSR(TrainingDataSet trainData, double minSingularVal, int maxSVDDim, int projDim, ProjType projType, boolean randStart)
	{

		this(trainData, minSingularVal, maxSVDDim, projDim, CPSR.DEF_MAX_HIST, projType,  randStart, CPSR.DEF_MAX_Test);
	}

	public MemEffCPSR(TrainingDataSet trainData, double minSingularVal, int maxSVDDim, int projDim, int maxHistLen, ProjType projType, boolean randStart, int maxTestLen)
	{
		super(trainData, minSingularVal, maxSVDDim, maxHistLen, maxTestLen);
		this.projType = projType;
		this.projDim = projDim;
		one = DoubleMatrix.ones(projDim + 1, 1);
		e = DoubleMatrix.zeros(projDim + 1, 1);
		e.put(0, 0, 1);
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
		this.colToAddCache = Collections.synchronizedMap(new MaxSizeHashMap<Integer, DoubleMatrix>(10000000, 10000000));
		this.rowToAddCache = Collections.synchronizedMap(new MaxSizeHashMap<Integer, DoubleMatrix>(10000000, 10000000));

		//initializing observable matrices, Σ_H Σ_TH
		if (hist==null|| th == null)
		{
//			hist = new DoubleMatrix(histories.size() + 1, 1);
			th = new DoubleMatrix(projDim, projDim + 1);
			hist = new DoubleMatrix(projDim + 1, 1);
		}
		int num = Param.incrementalThreadsForPSR * (trainData.getBatchNumber()+1);
		while (1000%num!=0)
		{
			num--;
		}
		int threadsNum = Math.min(Param.MaximumThreadsForComputingPSR, num);
		// filling in hist and th matrix 
		// fixing the length is because it only needs to iterate one batch of data
		estimateHistoryAndTHMatricesMultiThread(50);
//		estimateHistoryAndTHMatricesMultiThread_FullyRebuild(threadsNum);
		/*
		 *  normalize th, hist
		 *  CountNumberForHMats is the times of updating hist matrix
		 *  CountNumberForThMats is the times of updating th matrix
		 */
//		hist.mmuli(1.0/(CountNumberForHMats[0]));
//		th.mmuli(1.0/(CountNumberForTHMats[0]));
		System.out.println("Having Constructed TH Matrix!");
//		DoubleMatrix normth = th.mmul(1.0 / CountNumberForTHMats[0]);
//		DoubleMatrix normhist = hist.mmul(1.0 / CountNumberForHMats[0]);
		DoubleMatrix normth = th;
		DoubleMatrix normhist = hist;
		// Taking SVD of th and initialize aoMats
		svdResults = computeTruncatedSVD(normth, minSingularVal, maxDim);
		pseudoInverse = Solve.pinv(svdResults[0].mmul(normth));
		aoMats = Collections.synchronizedMap(new HashMap<ActionObservation, DoubleMatrix>());
//		CaoMats = Collections.synchronizedMap(new HashMap<ActionObservation, DoubleMatrix>());
//		PaoMats = Collections.synchronizedMap(new HashMap<ActionObservation, DoubleMatrix>());
		for(ActionObservation actOb : trainData.getValidActionObservationSet())
		{
			aoMats.put(actOb, new DoubleMatrix(pseudoInverse.getColumns(), pseudoInverse.getColumns()));
//			CaoMats.put(actOb, new DoubleMatrix(tests.size(), histories.size() + 1));
//			PaoMats.put(actOb, new DoubleMatrix(projDim, projDim + 1));
		}
		
		// Computing first part of equation 41
		constructAOMatricesMuliThread_FullyRebuild(threadsNum);
//		for (ActionObservation actob: CaoMats.keySet())
//		{
//			WriteDoubleMatrixToExcel(CaoMats.get(actob), actob.toString());
//		}
//		WriteDoubleMatrixToExcel(P_TH, "null history");
//		for (ActionObservation actob: PaoMats.keySet())
//		{
//			aoMats.put(actob, svdResults[0].mmul(PaoMats.get(actob)).mmul(pseudoInverse));
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
//		DoubleMatrix e = Solve.solveLeastSquares(Phi_hist, DoubleMatrix.ones(histories.size() + 1, 1));
//		System.out.println(Phi_hist.mmul(e));
		DoubleMatrix PQ = svdResults[0].mmul(normth);
		DoubleMatrix minf = Solve.solveLeastSquares(PQ.transpose(), normhist);
		mInf = new Minf(minf);
		pv = PredictionVector.BuildPredctiveVector(PQ.mmul(e));
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
			if (colToAddCache.containsKey(testIndex)) {
				colToAdd = colToAddCache.get(testIndex);
			} else {
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
				DoubleMatrix randHistVector;
				if (histIndex == 0) 
				{
					// if the history is null, when the agent randomly starts, the φ(hj) = [1,1,1,...].
					if (Param.isRandomInit)
					{
						randHistVector = DoubleMatrix.ones(projDim + 1, 1);
					}
					else
					{
						// When the agent starts from a special position, the φ(hj) = [1, 0, 0, 0...]
						randHistVector = DoubleMatrix.zeros(projDim + 1, 1);
						randHistVector.put(0, 0, 1);
					}
				} 
				else
				{
					// if the history is not null, generate a random vector and expand an extra zero.
					randHistVector = getRandomVector(projDim, projType, -1 * histIndex);
					if (Param.isRandomInit)
					{
						randHistVector = DoubleMatrix.concatVertically(DoubleMatrix.ones(1), randHistVector);
					}
					else
					{
						randHistVector = DoubleMatrix.concatVertically(DoubleMatrix.zeros(1), randHistVector);
					}
				}
				// Computing Sˆ{−1} VˆT φ(hj)
//				currRowToAdd = pseudoInverse.mmul(randHistVector);
				if (PaoMats != null)
				{
					PaoMats.get(actOb).addi(getRandomVector(projDim, projType, testIndex).mmul(randHistVector.transpose()));
				}
				currRowToAdd = randHistVector.transpose().mmul(pseudoInverse);
				rowToAddCache.put(histIndex, currRowToAdd);
			}
			// Taking outer product
//			for (int rowIndex = 0; rowIndex < colToAdd.getRows(); rowIndex++)
//			{
//				double currEntry = colToAdd.get(rowIndex, 0);
//				DoubleMatrix tmp = currRowToAdd.mmul(currEntry);
//				DoubleMatrix updated_row = aoCoeffecientsMat.getRow(rowIndex);
//				updated_row = updated_row.add(tmp.transpose());
//				aoCoeffecientsMat.putRow(rowIndex, updated_row);
//			}
			// way 2 implement outer
			DoubleMatrix tmp = colToAdd.mmul(currRowToAdd);
			aoCoeffecientsMat.addi(tmp);
			if (CaoMats != null)
			{
				CaoMats.get(actOb).put(testIndex, histIndex, CaoMats.get(actOb).get(testIndex, histIndex) + 1);
			}
			aoMats.put(actOb, aoCoeffecientsMat);
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
//			int row = colToAdd.getRows();
			
			// currRowToAdd is a random vector φ(hi) for hi
			DoubleMatrix currRowToAdd = null;
			if(hi == 0)
			{
				if (Param.isRandomInit)
				{
					currRowToAdd = DoubleMatrix.ones(projDim + 1, 1);
				}
				else
				{
					currRowToAdd = DoubleMatrix.zeros(projDim + 1, 1);
					currRowToAdd.put(0,0,1);
				}
			}
			else
			{
				currRowToAdd = getRandomVector(projDim, projType, -1 * hi);
				if (Param.isRandomInit)
				{
					currRowToAdd = DoubleMatrix.concatVertically(DoubleMatrix.ones(1), currRowToAdd);	
				}
				else
				{
					currRowToAdd = DoubleMatrix.concatVertically(DoubleMatrix.zeros(1), currRowToAdd);
				}
			}
			
			// Taking outer product for ΣˆT ,H
//			for(int rowIndex = 0; rowIndex < row; rowIndex++)
//			{
//				double currEntry = colToAdd.get(rowIndex, 0);
//				DoubleMatrix tmp = currRowToAdd.mmul(currEntry).transpose();
//				DoubleMatrix updatedRow = th.getRow(rowIndex);
//				updatedRow = updatedRow.add(tmp);
//				th.putRow(rowIndex, updatedRow);
//			}
			
			// way 2 implement outer
			DoubleMatrix tmp = colToAdd.mmul(currRowToAdd.transpose());
			th.addi(tmp);
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
		DoubleMatrix randHistVec;
		if(hi == 0)
		{
			if (Param.isRandomInit)
			{
				randHistVec = DoubleMatrix.ones(projDim + 1, 1);
			}
			else
			{
				randHistVec = DoubleMatrix.zeros(projDim + 1, 1);
				randHistVec.put(0,0,1);
			}
		}
		else
		{
			randHistVec = getRandomVector(projDim, projType, -1 * hi);
			if (Param.isRandomInit)
			{
				randHistVec = DoubleMatrix.concatVertically(DoubleMatrix.ones(1), randHistVec);
			}
			else
			{
				randHistVec = DoubleMatrix.concatVertically(DoubleMatrix.zeros(1), randHistVec);
			}
		}
		
		// if history is not null, update ΣˆH matrix
			// Count the times of updating the hist matrix
		synchronized (CountNumberForHMats)
		{
			CountNumberForHMats[0]++;
//				Phi_hist.putRow(hi, randHistVec.transpose());
		}
		hist.putColumn(0, hist.getColumn(0).add(randHistVec));
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
				Random rand = new Random(Param.getRandomSeed());
				int rows = randVec.getRows();
				switch (projType) {
				case Spherical:
					// setting entries to random numbers drawn from gaussian distrubution.
					for (int i = 0; i < rows; i++) {
//						randVec.put(i, 0, rand.nextGaussian() / Math.sqrt(((double) projDim)));
//						randVec.put(i, 0, rand.nextGaussian() / ((double) projDim));
						randVec.put(i, 0, rand.nextGaussian());
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
	protected void WriteDoubleMatrixToExcel(DoubleMatrix aoMats, String actob) throws IOException
	{
		FileWriter csvReader = new FileWriter(actob + ".csv");
		for (int rowid = 0; rowid < aoMats.getRows(); rowid++)
		{
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


