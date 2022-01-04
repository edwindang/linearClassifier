package learn.lc.core;

import learn.math.util.VectorOps;

public class LogisticClassifier extends LinearClassifier {
	
	public LogisticClassifier(double[] weights) {
		super(weights);
	}
	
	public LogisticClassifier(int ninputs) {
		super(ninputs);
	}
	
	/**
	 * A LogisticClassifier uses the logistic update rule
	 * (AIMA Eq. 18.8): w_i \leftarrow w_i+\alpha(y-h_w(x)) \times h_w(x)(1-h_w(x)) \times x_i 
	 */
	public void update(double[] x, double y, double alpha) {
		// This must be implemented by you
		// wi = wi + alpha(y-hw(x)) * hw(x) * (1-hw(x)) * xi
		for(int i = 0;i<x.length;i++){
			this.weights[i] = this.weights[i] + alpha*(y-eval(x)) * eval(x) * (1 - eval(x)) * x[i];
		}
	}
	
	/**
	 * A LogisticClassifier uses a 0/1 sigmoid threshold at z=0.
	 */
	public double threshold(double z) {
		double calc;
		double denom = (1 + Math.exp(-z));
		calc = (double)1/denom;
		return calc;
	}

}
