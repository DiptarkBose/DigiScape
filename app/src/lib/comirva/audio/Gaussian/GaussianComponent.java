package lib.comirva.audio.Gaussian;

import java.io.IOException;
import java.util.Random;

import lib.comirva.audio.CholeskyDecomposition;
import lib.comirva.audio.Matrix;
import lib.comirva.audio.PointList;




/**
 * <b>Gaussian Component</b>
 *
 * <p>Description: </p>
 * A <code>GaussianMixture</code> consists of several gaussian components, which
 * are modeled by this class. A gaussian component is a n-dimensional gaussian
 * distribution.
 *
 * @see comirva.audio.util.gmm.GaussianMixture
 * @author Klaus Seyerlehner
 * @version 1.1
 */
public final class GaussianComponent 
{
  //fields defining the gaussian componente
  private int dimension;
  private Matrix mean;
  private double componentWeight;
  private Matrix covariances;

  //implementation details (optimizations)
  private double coefficient;
  //private Matrix invCovariances;
  private CholeskyDecomposition chol;
  private Matrix invChol;

  private static Random rnd = new Random();
  /**
   * don't make the threshold too large, or the correction routine cannot
   * eliminate the points the corrupt component is focusing on.
   **/
  private static double SINGULARITY_DETECTION_THRESHOLD = 1.0E-20;


  /**
   * Creates a gaussian component and checks the component settings for
   * correctness.
   *
   * @param componentWeight double the weight of this component
   * @param mean Matrix the mean vector of this component
   * @param covariances Matrix the covariance matrix of this component
   * @throws IllegalArgumentException thrown if any invaild or incorrect
   *                                  settings were found
   */
  public GaussianComponent(double componentWeight, Matrix mean, Matrix covariances) throws IllegalArgumentException
  {
    //check parameters
    if(componentWeight < 0 || componentWeight > 1)
      throw new IllegalArgumentException("the weight of the component must be in the intervall [0,1];");

    if(mean == null || covariances == null)
      throw new IllegalArgumentException("mean and covariances must not be null values;");

    //set weight, mean and covarainces
    this.mean = mean;
    this.covariances = covariances;
    this.componentWeight = componentWeight;

    //set dimension of the component
    dimension = mean.getRowDimension();

    //check if the settings
    try
    {
      covariances.times(mean);

      //actualize optimization fields
      actualizeOptimizationFields();
    }
    catch(Exception e)
    {
      throw new IllegalArgumentException("mean and covariance matrix must have compatible shapes and the covarince matrix must not be singular;");
    }
  }


  /**
   * This constructor is for xml serialization
   */
  private GaussianComponent()
  {
  }


  /**
   * Computes the components weight or in other words the prior probability of
   * being generated by component i.<br>
   * <br>
   * [P(C = i)]
   *
   * @return double the components weight
   */
  public double getComponentWeight()
  {
    return componentWeight;
  }


  /**
   * Returns the probaility of drawing the given sample from this n-dimensional
   * gaussian distribution.<br>
   * <br>
   * [p(x | C = i)]
   *
   * @param x Matrix a sample
   * @return double the probability of this sample with respect to this
   *                distribution
   */
  /*public double getSampleProbability(Matrix x)
  {
    Matrix diff, result;
    double value;

    diff = x.minus(mean);
    result = diff.transpose();
    result = result.times(invCovariances);
    result = result.times(diff);
    value = -0.5d * result.get(0,0);
    return coefficient * Math.exp(value);
  }*/


  /**
   * Returns the probaility of drawing the given sample from this n-dimensional
   * gaussian distribution weighted with the prior probability of this
   * component.<br>
   * <br>
   * [p(x | C = i) * P(C = i)]<br>
   * <br >
   * So this is the probability of the sample under this gaussian distribution,
   * which is only a part of the wohl distribution. This is an optimized
   * implementation using the cholesky decomposition.
   *
   * @param x Matrix a sample
   * @return double the probability of this sample with respect to this
   *                distribution weighted with the prior probability of this
   *                component.
   */
  public double getWeightedSampleProbability(Matrix x)
  {
    Matrix diff, result;
    double value = 0;;
    double[] row;

    diff = x.minus(mean);
    result = diff.transpose();
    result = result.timesTriangular(invChol);

    row = result.getArray()[0];
    for(int i = 0; i < row.length; i++)
      value += row[i]*row[i];

    value *= -0.5d;
    return coefficient * Math.exp(value) * componentWeight;
  }


  /**
   * Returns the probaility of drawing the given sample from this n-dimensional
   * gaussian distribution weighted with the prior probability of this
   * component.<br>
   * <br>
   * [p(x | C = i) * P(C = i)]<br>
   * <br >
   * So this is the probability of the sample under this gaussian distribution,
   * which is only a part of the wohl distribution.
   *
   * @param x Matrix a sample
   * @return double the probability of this sample with respect to this
   *                distribution weighted with the prior probability of this
   *                component.
   *
   * Attention: SLOW!!!
   */
  /*public double getWeightedSampleProbability(Matrix x)
  {
    Matrix diff, result;
    double value = 0;;

    diff = x.minus(mean);
    result = diff.transpose();
    result = result.times(invCovariances);
    result = result.times(diff);
    value = -0.5d * result.get(0,0);

    return coefficient * Math.exp(value) * componentWeight;
  }*/


  /**
   * This method performs the maximization step for this component given the
   * sample points and the esimates <span>p_ij = P(C=i | x_j)</span> for sample
   * j of being generated by this component under the assumption that this
   * sample has been drawn from this GMM. So i is fixed to the index of this
   * component.
   *
   * @param samplePoints PointList the sample points
   * @param p_ij double[] the estimates
   *
   * @throws CovarianceSingularityException thrown if this component got
   *                                        singular druing the maximization step
   */
  protected void maximise(PointList samplePoints, double[] p_ij) throws CovarianceSingularityException
  {
    //compute new component weight [p_i = SUM over all j: P(C=i | x_j)]
    double p_i = 0;
    for(int j = 0; j < samplePoints.size(); j++)
      p_i += p_ij[j];

    //compute new mean
    Matrix mean = new Matrix(dimension, 1);
    for(int j = 0; j < samplePoints.size(); j++)
    {
      Matrix x = (Matrix) samplePoints.get(j);
      x = x.times(p_ij[j]);
      mean.plusEquals(x);
    }
    mean.timesEquals(1/p_i);

    //compute new covarince matrix
    Matrix covariances = new Matrix(dimension, dimension);
    for(int j = 0; j < samplePoints.size(); j++)
    {
      Matrix x = (Matrix) samplePoints.get(j);
      Matrix diff = x.minus(mean);
      diff = diff.times(diff.transpose());
      diff.timesEquals(p_ij[j]);
      covariances.plusEquals(diff);
    }
    covariances.timesEquals(1/p_i);

    //set covarince matrix
    this.covariances = covariances;
    //set new mean
    this.mean = mean;
    //set new component weight
    componentWeight = p_i/samplePoints.size();
    //actualize optimization fields
    actualizeOptimizationFields();
  }


  /**
   * Returns a sample drawn from this n-dimensional gaussian distribution.
   *
   * @return double[] the sample drawn from this distribution
   */
  public double[] nextSample()
  {
    //get standard normal vector
    double[] z = getStandardNormalVector(dimension);

    //transform back from standard normal to the given gaussian distribution
    Matrix Z = new Matrix(z, dimension);
    Matrix sample = chol.getL().times(Z);
    sample.plusEquals(mean);

    return sample.getColumnPackedCopy();
  }


  /**
   * Returns a vector, whose components are n independent standard normal
   * variates.
   *
   * @param dimension int the number of components (dimensionality) of the
   *                      vector
   * @return double[] a vector, whose components are standard normal variates
   */
  public static double[] getStandardNormalVector(int dimension)
  {
    double[] v = new double[dimension];
    for(int i = 0; i < v.length; i++)
      v[i] = rnd.nextGaussian();
    return v;
  }


  /**
   * Returns the dimension of this n-dimensional gaussian distribution.
   *
   * @return int number of dimensions
   */
  public int getDimension()
  {
    return this.dimension;
  }


  /**
   * Prints some information about this componet.
   * This is for debugging purpose only.
   */
  public void print()
  {
    System.out.println("mean:");
    mean.print(6, 3);
    System.out.println("covariance matrix:");
    covariances.print(6,3);
    System.out.println("component weight:");
    System.out.println(componentWeight);
    System.out.println("---------------------------------------------");
  }


  /**
   * For testing purpose only.
   *
   * @return Matrix the mean vector
   */
  public Matrix getMean()
  {
    return mean;
  }


  /**
   * Recomputes the optimization fields. The opimization fields store values,
   * that stay the same till the configuration of the component changes.
   * Consequently they don't have to be computed for each sample, but for each
   * configuration.
   *
   * @throws CovarianceSingularityException thrown if this component's
   *                                        covariance matrix got singular
   */
  private void actualizeOptimizationFields() throws CovarianceSingularityException
  {
    double detCovariances = 0.0d;

    //compute cholesky decomposition
    chol = covariances.chol();

    //root of the determinante is the product of all diagonal elements of the cholesky decomposition
    double[][] covars = chol.getL().getArray();
    detCovariances = 1.0d;
    for(int i = 0; i < covars.length; i++)
      detCovariances *= covars[i][i];
    detCovariances *= detCovariances;

    //is covariance matrix is singular?
    if(detCovariances < SINGULARITY_DETECTION_THRESHOLD)
      throw new CovarianceSingularityException(null);

    //get inverse of chol decomposition
    invChol = chol.getL().inverse();

    //invCovariances = covariances.inverse();
    coefficient = 1.0d / (Math.pow((2.0d * Math.PI), dimension/2.0) * Math.pow(detCovariances, 0.5d));
  }


  /**
   * Writes the xml representation of this object to the xml ouput stream.<br>
   * <br>
   * There is the convetion, that each call to a <code>writeXML()</code> method
   * results in one xml element in the output stream.
   *
   * @param writer XMLStreamWriter the xml output stream
   *
   * @throws IOException raised, if there are any io troubles
   * @throws XMLStreamException raised, if there are any parsing errors
   */
  /*public void writeXML(XMLStreamWriter writer) throws IOException, XMLStreamException
  {
    writer.writeStartElement("component");
    writer.writeAttribute("weight", Double.toString(componentWeight));

    mean.writeXML(writer);
    covariances.writeXML(writer);

    writer.writeEndElement();
  }
*/

  /**
   * Reads the xml representation of an object form the xml input stream.<br>
   * <br>
   * There is the convention, that <code>readXML()</code> starts parsing by
   * checking the start tag of this object and finishes parsing by checking the
   * end tag. The caller has to ensure, that at method entry the current token
   * is the start tag. After the method call it's the callers responsibility to
   * move from the end tag to the next token.
   *
   * @param parser XMLStreamReader the xml input stream
   *
   * @throws IOException raised, if there are any io troubles
   * @throws XMLStreamException raised, if there are any parsing errors
   */
  /*public void readXML(XMLStreamReader parser) throws IOException, XMLStreamException
  {
    parser.require(XMLStreamReader.START_ELEMENT, null, "component");
    componentWeight = Double.parseDouble(parser.getAttributeValue(null, "weight"));
    parser.next();

    mean = new Matrix(0,0);
    mean.readXML(parser);
    parser.next();

    covariances = new Matrix(0,0);
    covariances.readXML(parser);
    parser.next();

    dimension = mean.getRowDimension();

    try
    {
      actualizeOptimizationFields();
    }
    catch (CovarianceSingularityException ex)
    {
      System.out.println(ex.toString());
    }


    parser.require(XMLStreamReader.END_ELEMENT, null, "component");
  }
*/

  /**
   * This method allows to read a gaussian component from a xml input stream as
   * recommended by the XMLSerializable interface.
   *
   * @see comirva.audio.XMLSerializable
   * @param parser XMLStreamReader the xml input stream
   * @return GaussianMixture the GMM read from the xml stream
   *
   * @throws IOException raised, if there are any io troubles
   * @throws XMLStreamException raised, if there are any parsing errors
   */
 /* public static GaussianComponent readGC(XMLStreamReader parser) throws IOException, XMLStreamException
  {
    GaussianComponent gc = new GaussianComponent();
    gc.readXML(parser);
    return gc;
  }
*/}
