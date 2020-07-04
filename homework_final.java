import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.File;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;
import java.util.TreeMap;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class KnnPattern
{
    //��double���ͺ�string���ͼ���������һ���࣬����double��ʾΪ���������������Լ�
    //�������ľ��룬string��ʾ���Լ���������ģ�ͣ������м����ĳ������͡�
		public static class DoubleString implements WritableComparable<DoubleString>
		{
			private Double distance = 0.0;
			private String model = null;

			public void set(Double lhs, String rhs)
			{
				distance = lhs;
				model = rhs;
			}
			public Double getDistance()
				return distance;

			public String getModel()
				return model;

			@Override
			public void readFields(DataInput in) throws IOException
			{
				distance = in.readDouble();
				model = in.readUTF();
			}

			@Override
			public void write(DataOutput out) throws IOException
			{
				out.writeDouble(distance);
				out.writeUTF(model);
			}

			@Override
			public int compareTo(DoubleString o)
			{
				return (this.model).compareTo(o.model);
			}
		}
    //Mapper�࣬���ڶ����ݽ���ӳ��
	public static class KnnMapper extends Mapper<Object, Text, NullWritable, DoubleString>
	{
		DoubleString distanceAndModel = new DoubleString();
		TreeMap<Double, String> KnnMap = new TreeMap<Double, String>();

		int K;

		double normalisedSAge;
		double normalisedSIncome;
		String sStatus;
		String sGender;
		double normalisedSChildren;

		//�ֶ�ȡ�������еĸ��������Сֵ
		double minAge = 18;
		double maxAge = 77;
		double minIncome = 5000;
		double maxIncome = 67789;
		double minChildren = 0;
		double maxChildren = 5;

		//������Ȩ�أ�ȡֵ0-1
		private double normalisedDouble(String n1, double minValue, double maxValue)
		{
			return (Double.parseDouble(n1) - minValue) / (maxValue - minValue);
		}

		//����ֻ������ȡֵ�ı����ĶԱ�Ȩ��
		private double nominalDistance(String t1, String t2)
		{
			if (t1.equals(t2))
			{
				return 0;
			}
			else
			{
				return 1;
			}
		}

		//Ϊ����׼����Ȩ��ȡһ��ƽ��
		private double squaredDistance(double n1)
		{
			return Math.pow(n1,2);
		}

		//�����ܵġ����롱
		private double totalSquaredDistance(double R1, double R2, String R3, String R4, double R5, double S1,
				double S2, String S3, String S4, double S5)
		{
			double ageDifference = S1 - R1;
			double incomeDifference = S2 - R2;
			double statusDifference = nominalDistance(S3, R3);
			double genderDifference = nominalDistance(S4, R4);
			double childrenDifference = S5 - R5;

			return squaredDistance(ageDifference) + squaredDistance(incomeDifference) + statusDifference + genderDifference + squaredDistance(childrenDifference);
		}

		//����setup
		@Override
		protected void setup(Context context) throws IOException, InterruptedException
		{
			if (context.getCacheFiles() != null && context.getCacheFiles().length > 0)
			{
				String knnParams = FileUtils.readFileToString(new File("./knnParamFile"));
				StringTokenizer st = new StringTokenizer(knnParams, ",");

				K = Integer.parseInt(st.nextToken());
				normalisedSAge = normalisedDouble(st.nextToken(), minAge, maxAge);
				normalisedSIncome = normalisedDouble(st.nextToken(), minIncome, maxIncome);
				sStatus = st.nextToken();
				sGender = st.nextToken();
				normalisedSChildren = normalisedDouble(st.nextToken(), minChildren, maxChildren);
			}
		}

		//����map�������ݽ���ӳ�䴦��
		@Override
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException
		{
			String rLine = value.toString();
			StringTokenizer st = new StringTokenizer(rLine, ",");

			double normalisedRAge = normalisedDouble(st.nextToken(), minAge, maxAge);
			double normalisedRIncome = normalisedDouble(st.nextToken(), minIncome, maxIncome);
			String rStatus = st.nextToken();
			String rGender = st.nextToken();
			double normalisedRChildren = normalisedDouble(st.nextToken(), minChildren, maxChildren);
			String rModel = st.nextToken();

			double tDist = totalSquaredDistance(normalisedRAge, normalisedRIncome, rStatus, rGender,
					normalisedRChildren, normalisedSAge, normalisedSIncome, sStatus, sGender, normalisedSChildren);

			KnnMap.put(tDist, rModel);
			if (KnnMap.size() > K)
			{
				KnnMap.remove(KnnMap.lastKey());
			}
		}

		//����һ���������
		@Override
		protected void cleanup(Context context) throws IOException, InterruptedException
		{
			for(Map.Entry<Double, String> entry : KnnMap.entrySet())
			{
				  Double knnDist = entry.getKey();
				  String knnModel = entry.getValue();
				  distanceAndModel.set(knnDist, knnModel);
				  context.write(NullWritable.get(), distanceAndModel);
			}
		}
	}

	//reduer��
	public static class KnnReducer extends Reducer<NullWritable, DoubleString, NullWritable, Text>
	{
		TreeMap<Double, String> KnnMap = new TreeMap<Double, String>();
		int K;

		//����setup�ĺ���
		@Override
		protected void setup(Context context) throws IOException, InterruptedException
		{
			if (context.getCacheFiles() != null && context.getCacheFiles().length > 0)
			{
				String knnParams = FileUtils.readFileToString(new File("./knnParamFile"));
				StringTokenizer st = new StringTokenizer(knnParams, ",");
				K = Integer.parseInt(st.nextToken());
			}
		}

		//reduce�������Խ���map������ݽ��й�Լ���ó���������ж�
		@Override
		public void reduce(NullWritable key, Iterable<DoubleString> values, Context context) throws IOException, InterruptedException
		{
			for (DoubleString val : values)
			{
				String rModel = val.getModel();
				double tDist = val.getDistance();

				KnnMap.put(tDist, rModel);
				if (KnnMap.size() > K)
				{
					KnnMap.remove(KnnMap.lastKey());
				}
			}

				List<String> knnList = new ArrayList<String>(KnnMap.values());
				Map<String, Integer> freqMap = new HashMap<String, Integer>();

			    for(int i=0; i< knnList.size(); i++)
			    {
			        Integer frequency = freqMap.get(knnList.get(i));
			        if(frequency == null)
			        {
			            freqMap.put(knnList.get(i), 1);
			        } else
			        {
			            freqMap.put(knnList.get(i), frequency+1);
			        }
			    }
			    String mostCommonModel = null;
			    int maxFrequency = -1;
			    for(Map.Entry<String, Integer> entry: freqMap.entrySet())
			    {
			        if(entry.getValue() > maxFrequency)
			        {
			            mostCommonModel = entry.getKey();
			            maxFrequency = entry.getValue();
			        }
			    }
			context.write(NullWritable.get(), new Text(mostCommonModel));
		}
	}

	//main���������ڽ��ܲ������е���
	public static void main(String[] args) throws Exception
	{
		Configuration conf = new Configuration();

		if (args.length != 3)
		{
			System.err.println("Usage: KnnPattern <in> <out> <parameter file>");
			System.exit(2);
		}

		Job job = Job.getInstance(conf, "Find K-Nearest Neighbour");
		job.setJarByClass(KnnPattern.class);
		job.addCacheFile(new URI(args[2] + "#knnParamFile"));

		job.setMapperClass(KnnMapper.class);
		job.setReducerClass(KnnReducer.class);
		job.setNumReduceTasks(1);

		job.setMapOutputKeyClass(NullWritable.class);
		job.setMapOutputValueClass(DoubleString.class);
		job.setOutputKeyClass(NullWritable.class);
		job.setOutputValueClass(Text.class);

		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));

		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
}
