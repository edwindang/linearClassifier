package learn.nn.core;

/**
 * A PerceptronUnit is a Unit that uses a hard threshold
 * activation function.
 */
public class PerceptronUnit extends NeuronUnit {
	
	/**
	 * The activation function for a Perceptron is a hard 0/1 threshold
	 * at z=0. (AIMA Fig 18.7)
	 */
	@Override
	public double activation(double z) {
		// This must be implemented by you
		if(z >= 0){
			return 1.0;
		}
		else{
			return 0.0;
		}
	}
	
	/**
	 * Update this unit's weights using the Perceptron learning
	 * rule (AIMA Eq 18.7).
	 * Remember: If there are n input attributes in vector x,
	 * then there are n+1 weights including the bias weight w_0. 
	 */
	@Override
	public void update(double[] x, double y, double alpha) {
		// This must be implemented by you
		int length = x.length;
		for(int i = 0;i<length;i++){
			double ans = getWeight(i) + alpha * (y-h_w(x)) * x[i];
			setWeight(i, ans);
		}
	}
}
