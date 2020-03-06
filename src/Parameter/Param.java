package Parameter;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.json.simple.JSONObject;
import com.google.gson.Gson;
import com.google.gson.JsonIOException;
import com.google.gson.JsonSyntaxException;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
public class Param {
	public static int num_atoms = 21;
	public static Double vmin = -2000.0;
	public static Double vmax = 200.0;
	public static Double discount = 0.95;
	public static boolean C51 = true;
	public static boolean Quantile_DRL = false;
	public static String initialized_type = "uniform"; // uniform, gaussian
	public static int left_bound = 0;
	public static int right_bound = num_atoms;
	public static String Update = "update";
	public static Double Learning_rate = 0.001;
	public static Double Huber_Loss_K = 0.5;
	public static boolean loading = false;
	public static String update_pv = "numerator";
	public static int numThreadsForComputingTargets = 5;
	public static int seed = 10;
	public static double tao_uniform_loss = 0.2;
	public static double variance_part = 0.1;
	public static int actionSize = -1;
	public static int observationSize = -1;
	public static int rewardSize = -1;
	public static int trees = 10;
	public static int SVD_DIM = 10;
	public static Integer[] randSeed = new Integer[10000000];
	public static boolean isRandomInit = false;
	public static String RunningVersion;
	public static int FinalThreshold = -1;
	public static int WeakThreshold = -1;
	public static boolean TruncatedLikelihoods = true;
	public static boolean introducedReward = true;
	public static boolean NormalizePredictedResult = true;
	public static boolean StartingAtSpecialPosition = true;
	public static String GameName;
	private static int count = 0;
	public static int MaximumQIteration = 50;
	public static int MaximumThreadsForComputingQ = 100;
	public static int MaximumThreadsForComputingPSR = 100;
	public static int incrementalThreadsForPSR = 10;
	public static String planningType = "Action";
	public static String POMDPAction = "Policy";
	public static void initialRandom(int offset)
	{
		for (int i = 0; i < randSeed.length; i++)
		{
			randSeed[i] = i;
		}
		Collections.shuffle(Arrays.asList(randSeed), new Random(offset*100));
	}
	public static int getRandomSeed()
	{
		return randSeed[count++];
	}
	public static void read_Param()
	{
        JSONParser jsonParser = new JSONParser();
		FileReader reader;
		Gson gson = new Gson();
		try {
			reader = new FileReader("Param.json");
			JSONObject obj = (JSONObject)jsonParser.parse(reader);
	    	Param.num_atoms = gson.fromJson(obj.get("num_atoms").toString(), Integer.class);
			Param.vmin = gson.fromJson(obj.get("vmin").toString(), Double.class);
			Param.vmax = gson.fromJson(obj.get("vmax").toString(), Double.class);
			Param.discount = gson.fromJson(obj.get("discount").toString(), Double.class);
			Param.C51 = gson.fromJson(obj.get("C51").toString(), boolean.class);
			Param.initialized_type = gson.fromJson(obj.get("initialized_type").toString(), String.class); // uniform, gaussian
			Param.left_bound = gson.fromJson(obj.get("left_bound").toString(), Integer.class);
			Param.right_bound = gson.fromJson(obj.get("right_bound").toString(), Integer.class);
			Param.Update = gson.fromJson(obj.get("Update").toString(), String.class);
			Param.Quantile_DRL = gson.fromJson(obj.get("Quantile_DRL").toString(), boolean.class);
			Param.Learning_rate = gson.fromJson(obj.get("Learning_rate").toString(), Double.class);
			Param.Huber_Loss_K = gson.fromJson(obj.get("Huber_Loss_K").toString(), Double.class);
			Param.loading = gson.fromJson(obj.get("Loading").toString(), boolean.class);
			Param.update_pv = gson.fromJson(obj.get("Update_PV").toString(), String.class);
			Param.numThreadsForComputingTargets = gson.fromJson(obj.get("numThreadsForComputingTargets").toString(), Integer.class);
			Param.seed = gson.fromJson(obj.get("seed").toString(), Integer.class);
			Param.tao_uniform_loss = gson.fromJson(obj.get("tao_uniform_loss").toString(), Double.class);
			Param.variance_part = gson.fromJson(obj.get("variance_part").toString(), Double.class);
			Param.isRandomInit = gson.fromJson(obj.get("RandomInit").toString(), boolean.class);
			Param.introducedReward = gson.fromJson(obj.get("IntroduceReward").toString(), boolean.class);
			Param.TruncatedLikelihoods = gson.fromJson(obj.get("TruncatedLikelihoods").toString(), boolean.class);
			Param.MaximumQIteration = gson.fromJson(obj.get("QIterations").toString(), Integer.class);
//			Param.planningType = gson.fromJson(obj.get("planningType").toString(), String.class);
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (JsonIOException e) {
		// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (JsonSyntaxException e) {
		// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ParseException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
