package learn.lc.core;

import java.util.Arrays;

import learn.math.util.VectorOps;

public class PerceptronClassifier extends LinearClassifier {
	
	public PerceptronClassifier(double[] weights) {
		super(weights);
	}
	
	public PerceptronClassifier(int ninputs) {
		super(ninputs);
	}
	
	/**
	 * A PerceptronClassifier uses the perceptron learning rule
	 * (AIMA Eq. 18.7): w_i \leftarrow w_i+\alpha(y-h_w(x)) \times x_i 
	 */
	public void update(double[] x, double y, double alpha) {
		// This must be implemented by you
		// wi =  wi + alpha(y-hw(x)) * xi
		int length = x.length;
		for(int i = 0;i<length;i++){
			this.weights[i] = this.weights[i] + alpha * (y-eval(x)) * x[i];
		}
	}
	
	/**
	 * A PerceptronClassifier uses a hard 0/1 threshold.
	 */
	public double threshold(double z) {
		// This must be implemented by you
		if(z >= 0){
			return 1.0;
		}
		else{
			return 0.0;
		}
	}
	
}
