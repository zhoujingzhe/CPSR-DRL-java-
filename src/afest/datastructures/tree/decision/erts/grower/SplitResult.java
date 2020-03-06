package afest.datastructures.tree.decision.erts.grower;

import java.io.Serializable;
import java.util.ArrayList;

import afest.datastructures.tree.decision.erts.ERTSplit;
import afest.datastructures.tree.decision.interfaces.ITrainingPoint;

public class SplitResult <R extends Serializable, O extends Serializable, T extends ITrainingPoint<R, O>>{
	public ERTSplit<R> BestSplit;
	public ArrayList<T> LeftResult;
	public ArrayList<T> RightResult;
	public double score = -1;
}
