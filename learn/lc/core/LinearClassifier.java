package learn.lc.core;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Random;

import learn.math.util.VectorOps;

abstract public class LinearClassifier {
	
	public double[] weights;
	
	public LinearClassifier(double[] weights) {
		this.weights = weights;
	}
	
	public LinearClassifier(int ninputs) {
		this.weights = new double[ninputs];
	}
	
	/**
	 * Update the weights of this LinearClassifer using the given
	 * inputs/output example and learning rate alpha.
	 */
	abstract public void update(double[] x, double y, double alpha);

	/**
	 * Threshold the given value using this LinearClassifier's
	 * threshold function.
	 */
	abstract public double threshold(double z);

	/**
	 * Evaluate the given input vector using this LinearClassifier
	 * and return the output value.
	 * This value is: Threshold(w \cdot x)
	 */
	public double eval(double[] x) {
		return threshold(VectorOps.dot(this.weights, x));
	}
	
	/**
	 * Train this LinearClassifier on the given Examples for the
	 * given number of steps, using given learning rate schedule.
	 * ``Typically the learning rule is applied one example at a time,
	 * choosing examples at random (as in stochastic gradient descent).''
	 * See AIMA p. 724.
	 * @throws IOException 
	 */
	public void train(List<Example> examples, int nsteps, LearningRateSchedule schedule, int choice, String argument) throws IOException {
		Random random = new Random();
		int n = examples.size();
		FileWriter writer = null;
		if (choice==0) {
			try {
				File myObj = new File("perceptron_"+argument);
				if (myObj.createNewFile()) {
					System.out.println("File created: " + myObj.getName());
				} else {
					System.out.println("File already exists.");
				}
			}
			catch (IOException e) {
				System.out.println("Error in creating file");
				e.printStackTrace();
			}
			try {
				FileWriter perceptronWriter = new FileWriter("perceptron_"+argument);
				writer = perceptronWriter;
			}
			catch (IOException e) {
				System.out.println("Unable to write to file");
				e.printStackTrace();
			}
		}
		
		else if (choice==1) {
			try {
				File myObj = new File("logistic_"+argument);
				if (myObj.createNewFile()) {
					System.out.println("File created: " + myObj.getName());
				} else {
					System.out.println("File already exists.");
				}
			}
			catch (IOException e) {
				System.out.println("Error in creating file");
				e.printStackTrace();
			}
			try {
				FileWriter logisticWriter = new FileWriter("logistic_"+argument);
				writer = logisticWriter;
			} catch (IOException e) {
				System.out.println("Unable to write to file");
				e.printStackTrace();
			}
		}
		for (int i=1; i <= nsteps; i++) {
			int j = random.nextInt(n);
			Example ex = examples.get(j);
			this.update(ex.inputs, ex.output, schedule.alpha(i));
			this.trainingReport(examples, i, nsteps, choice, writer);
		}
		writer.close();
		
	}

	/**
	 * Train this LinearClassifier on the given Examples for the
	 * given number of steps, using given constant learning rate.
	 * @throws IOException 
	 */
	public void train(List<Example> examples, int nsteps, double constant_alpha, int choice, String argument) throws IOException {
		if (constant_alpha==0.0) {
			train(examples, nsteps, new DecayingLearningRateSchedule() {
			}, choice, "decaying_alpha_"+argument);
		}
		else {
			train(examples, nsteps, new LearningRateSchedule() {
				public double alpha(int t) { return constant_alpha; }
			}, choice, "constant_alpha_"+argument);
		}
	}
	
	/**
	 * This method is called after each weight update during training.
	 * Subclasses can override it to gather statistics or update displays.
	 */
	protected void trainingReport(List<Example> examples, int stepnum, int nsteps, int choice, FileWriter writer) {
		//System.out.println(stepnum + "\t" + squaredErrorPerSample(examples));
		if (choice==0) {
			try {
				writer.write(stepnum + " " + accuracy(examples) + "\n");
			}
			catch (IOException e) {
				System.out.println("Unable to write to file");
				e.printStackTrace();
			}
		}
		else if (choice==1) {
			try {
				writer.write(stepnum + " " + squaredErrorPerSample(examples) + "\n");
			} catch (IOException e) {
				System.out.println("Unable to write to file");
				e.printStackTrace();
			}
		}
	}
	
	
	/**
	 * Return the squared error per example (Mean Squared Error) for this
	 * LinearClassifier on the given Examples.
	 * The Mean Squared Error is the total L_2 loss divided by the number
	 * of samples.
	 */
	public double squaredErrorPerSample(List<Example> examples) {
		double sum = 0.0;
		for (Example ex : examples) {
			double result = eval(ex.inputs);
			double error = ex.output - result;
			sum += error*error;
		}
		return 1-(sum / examples.size());
	}

	/**
	 * Return the proportion of the given Examples that are classified
	 * correctly by this LinearClassifier.
	 * This is probably only meaningful for classifiers that use
	 * a hard threshold. Use with care.
	 */
	public double accuracy(List<Example> examples) {
		int ncorrect = 0;
		for (Example ex : examples) {
			double result = eval(ex.inputs);
			if (result == ex.output) {
				ncorrect += 1;
			}
		}
		return (double)ncorrect / examples.size();
	}

}
