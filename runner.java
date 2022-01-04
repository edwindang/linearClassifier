import learn.lc.core.*;
import java.util.*;
import java.io.*;

public class runner {
	public static void main(String[] args) throws IOException {
		String argument = args[0];
		File data = new File(argument);
		List<Example> examples = new ArrayList<>();
		int ninputs = 0;
		Scanner first = new Scanner(System.in);
		System.out.println("Do you wish to run perceptron classifier or logistic classifier? (Enter 'p' or 'l')");
		char ans = first.next().charAt(0);
		Double rate = null;
		int run = -1;
		if (ans=='p') {
			run = 0;
		}
		else if (ans=='l') {
			run = 1;
		}
		else {
			System.out.println("Error. Please provide valid input.");
		}
		System.out.println("Do you want to use a fixed or decaying learning rate alpha? (Enter 'f' or 'd')");
		char alpha = first.next().charAt(0);
		if (alpha=='f') {
			System.out.println("Enter your fixed learning rate: ");
			rate = first.nextDouble();
			if (rate<=0.0) {
				System.out.println("Error. Please enter a learning rate greater than zero.");
				rate = null;
			}
		}
		else if (alpha=='d') {
			rate = 0.0;
		}
		else {
			System.out.println("Invalid input. Please try again.");
		}
		System.out.println("How many runs do you want to use?");
		int runs = first.nextInt();
		first.close();
		try {
			Scanner s = new Scanner(data);
			//if (argument.contains("earthquake")) {
				while (s.hasNextLine()) {
					String line = s.nextLine();
					String[] items = line.split(",");
					ninputs = items.length-1;
					double[] input = new double[items.length-1];
					double output;
					for (int i = 0; i<items.length-1; i++) {
						input[i] = Double.valueOf(items[i]);
					}
					output = Double.valueOf(items[items.length-1]);
					Example ex = new Example(input, output);
					examples.add(ex);
				}
				s.close();
			
		}
		catch (FileNotFoundException e) {
			System.out.println("Error occured in file retrieval");
			e.printStackTrace();
		}
		
		if (run==0) {
			PerceptronClassifier perceptron = new PerceptronClassifier(ninputs);
			perceptron.train(examples, runs, rate, run, argument);
			System.out.println("Weights:");
			for (double d: perceptron.weights) {
				System.out.println(d);
			}
		}
		else if (run==1) {
			LogisticClassifier logistic = new LogisticClassifier(ninputs);
			logistic.train(examples, runs, rate, run, argument);
			System.out.println("Weights:");
			for (double d: logistic.weights) {
				System.out.println(d);
			}
		}
	}

}
