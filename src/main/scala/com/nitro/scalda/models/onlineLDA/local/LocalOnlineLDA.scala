package com.nitro.scalda.models.onlineLDA.local

import java.io._

import breeze.linalg.{ DenseMatrix, sum }
import breeze.numerics.{ abs, exp }
import breeze.stats.distributions.{ Gamma => G }
import breeze.stats.mean
import com.nitro.scalda.evaluation.perplexity.Perplexity._
import com.nitro.scalda.Utils
import com.nitro.scalda.models._
import com.nitro.scalda.tokenizer.StanfordLemmatizer

import scala.util.Try

object LocalOnlineLda {

  def apply(p: OnlineLdaParams) =
    new LocalOnlineLda(p)

  // If you are only wanting to infer topic proportions,
  // parameters can be set to default
  lazy val empty = apply(
    OnlineLdaParams(
      vocabulary = IndexedSeq.empty[String],
      alpha = 0.0,
      eta = 0.0,
      decay = 1.0,
      learningRate = 0.0,
      maxIter = 100,
      convergenceThreshold = 0.001,
      numTopics = 0,
      totalDocs = 0
    )
  )
}

/**
 * Local, non-distributed version of the online LDA algorithm
 *
 * @param params online LDA parameters.  These are static parameters
 */
class LocalOnlineLda(params: OnlineLdaParams) extends OnlineLda {

  override type BowMinibatch = Seq[Document]
  override type MinibatchSStats = MbSStats[DenseMatrix[Double], DenseMatrix[Double]]
  override type LdaModel = ModelSStats[DenseMatrix[Double]]
  override type Lambda = DenseMatrix[Double]
  override type Minibatch = Seq[String]

  /**
   * Process a minibatch of documents to infer topic proportions and "intermediate" topics.
   *
   * @param mb Minibatch of documents.
   * @param lambda Parameter that is a function of overall topics learned so far.
   * @return Sufficient statistics from this minibatch.
   */
  override def eStep(
    mb: BowMinibatch,
    lambda: Lambda,
    gamma: Gamma
  ): MinibatchSStats = {

    val numDocs = mb.size
    val eLogBeta = Utils.dirichletExpectation(lambda)
    val expELogBeta = exp(eLogBeta)
    val numTopics = expELogBeta.rows

    val mbTopicMatrix = DenseMatrix.zeros[Double](numTopics, expELogBeta.cols)
    val expELogTheta = exp(Utils.dirichletExpectation(gamma))

    //iterate through documents in minibatch
    for {
      (doc, docId) <- mb.zipWithIndex
    } {

      val ids = doc.wordIds
      val cts = DenseMatrix(doc.wordCts.map(_.toDouble))

      var gammaDoc = gamma(docId, ::).t.toDenseMatrix
      var expELogThetaDoc = expELogTheta(docId, ::).t.toDenseMatrix
      val expELogBetaDoc = expELogBeta(::, ids).toDenseMatrix
      var phiNorm = expELogThetaDoc * expELogBetaDoc + 1e-100

      var convergence = false
      var numIter = 0

      //Iterate until variational parameters converge
      while ((numIter < params.maxIter) && !convergence) {
        val lastGammaD = gammaDoc
        gammaDoc = (expELogThetaDoc :* (cts / phiNorm.t) * expELogBetaDoc.t) + params.alpha
        expELogThetaDoc = exp(Utils.dirichletExpectation(gammaDoc))
        phiNorm = expELogThetaDoc * expELogBetaDoc + 1e-100
        if (mean(abs(gammaDoc - lastGammaD)) < params.convergenceThreshold) convergence = true
        numIter += 1
      }

      gamma(docId, ::) := gammaDoc.toDenseVector.t

      val docTopicUpdate = expELogThetaDoc.t * (cts / phiNorm)

      for ((id, idx) <- ids.zipWithIndex) {
        mbTopicMatrix(::, id) :+= docTopicUpdate(::, idx)
      }

    }

    MbSStats(
      mbTopicMatrix :* expELogBeta,
      gamma,
      numDocs
    )
  }

  /**
   * Step merging overall topics with "intermediate" topics learned from the last minibatch.
   *
   * @param model current overall topics learned so far.
   * @param mbSStats sufficient statistics from the last minibatch ("intermediate topics").
   * @return An updated model.
   */
  override def mStep(
    model: LdaModel,
    mbSStats: MinibatchSStats
  ): LdaModel = {

    val rho = math.pow(params.decay + model.numUpdates, -params.learningRate)

    //merge overall topics with minibatch updates
    val updatedLambda = (model.lambda * (1 - rho)) + rho * ((mbSStats.topicUpdates * (params.totalDocs.toDouble / mbSStats.mbSize)) + params.eta)

    val numProcessed = model.numUpdates + 1

    model.copy(
      lambda = updatedLambda,
      numUpdates = numProcessed
    )
  }

  /**
   * Learn a topic model from a minibatch iterator over a corpus of documents.
   *
   * @return A learned LDA topic model.
   */
  def inference(minibatchIterator: Iterator[Minibatch]): LdaModel = {

    //Initialize model settings
    val vocabMapping = Utils.mapVocabId(params.vocabulary)

    val lambda = new DenseMatrix[Double](
      params.numTopics,
      vocabMapping.size,
      G(100.0, 1.0 / 100.0)
      .sample(params.numTopics * vocabMapping.size)
      .toArray
    )

    val lemmatizer =
      if (params.lemmatize)
        Some(StanfordLemmatizer())
      else
        None

    // mutable state
    var counter = 0
    var curModel = ModelSStats[DenseMatrix[Double]](lambda, vocabMapping, 0)
    // ^^

    //iterate through each minibatch
    while (minibatchIterator.hasNext) {

      counter += 1
      if (counter % 5 == 0) {
        println(s"$counter minibatches processed!")
      }

      //put minibatch in BOW form
      val rawMinibatch = minibatchIterator.next()
      val bowMinibatch = rawMinibatch.map { doc =>
        Utils.toBagOfWords(doc, vocabMapping, lemmatizer)
      }

      val initialGamma = new DenseMatrix[Double](
        bowMinibatch.size,
        params.numTopics,
        G(100.0, 1.0 / 100.0)
        .sample(bowMinibatch.size * params.numTopics)
        .toArray
      )

      //perform e-step
      val mbSStats = eStep(bowMinibatch, curModel.lambda, initialGamma)

      //compute perplexity of next minibatch if enabled
      if (params.perplexity) {
        val score = perplexity(
          bowMinibatch,
          mbSStats.topicProportions,
          curModel.lambda,
          params
        )

        val mbWordCt = sum(bowMinibatch.flatMap(_.wordCts))
        val wordScale = (score * bowMinibatch.size) / (params.totalDocs * mbWordCt.toDouble)

        println(s"per-word perplexity: ${exp(-wordScale)}")
      }

      //update current model with m-step
      curModel = mStep(curModel, mbSStats)
    }

    curModel
  }

  /**
   * Print top 10 words by probability from each of the topics of a given topic model.
   *
   * @param model A learned topic model
   */
  def printTopics(model: LdaModel): Unit = {

    val reverseVocab = model.vocabMapping.map(_.swap)
    val lambda = model.lambda

    //iterate through topics and print high probability words
    for {
      t <- 0 until lambda.rows
    } {

      val topic = lambda(t, ::).t

      val normalizer = sum(topic)

      val topTopics = topic
        .toArray
        .map { _ / normalizer }
        .zipWithIndex
        .sortBy { case (prob, _) => -prob }
        .map { case (prob, wordId) => (reverseVocab(wordId), prob) }
        .take(10)

      println(s"""Topic #$t: ${topTopics.toSeq.mkString(", ")}""")
    }
  }

  /**
   * Given a document and a topic model, infer the proportions of each topic in the document.
   *
   * @param doc A document.
   * @param model A learned topic model.
   * @return The topic proportions for the given document.
   */
  def topicProportions(
    doc: String,
    model: LdaModel,
    lemmatizer: Option[StanfordLemmatizer] = None
  ): Array[(Double, Int)] = {

    lazy val expElogBeta = exp(Utils.dirichletExpectation(model.lambda))

    val bowDoc = Utils.toBagOfWords(doc, model.vocabMapping, lemmatizer)

    val initialGamma = new DenseMatrix[Double](
      1,
      params.numTopics,
      G(100.0, 1.0 / 100.0).sample(params.numTopics).toArray
    )

    val docSStats = eStep(Seq(bowDoc), expElogBeta, initialGamma)

    val normalizer = sum(docSStats.topicProportions)

    docSStats.topicProportions(0, ::)
      .t
      .toArray
      .map(_ / normalizer)
      .zipWithIndex
  }

  /**
   * Compute the similarity between two documents by computing the Euclidean
   * distance between their topic proportions.
   */
  def documentSimilarity(
    doc1: String,
    doc2: String,
    model: LdaModel
  ): Double = {

    val lemmatizer = params.lemmatize match {
      case true => Some(StanfordLemmatizer())
      case _ => None
    }

    //set to zero if small
    val doc1TopicProportions = topicProportions(doc1, model, lemmatizer)
      .map(_._1)
      .map(x => if (x < 0.1) 0.0 else x)

    val doc2TopicProportions = topicProportions(doc2, model, lemmatizer)
      .map(_._1)
      .map(x => if (x < 0.1) 0.0 else x)

    Utils.euclideanDistance(doc1TopicProportions, doc2TopicProportions)
  }

  /**
   * Save a learned topic model. Uses Java object serialization.
   */
  def saveModel(model: LdaModel, saveLocation: File): Try[Unit] =
    Try {
      val oos = new ObjectOutputStream(new FileOutputStream(saveLocation))
      oos.writeObject(model)
      oos.close()
    }

  /**
   * Load a saved topic model from save location.
   * Uses Java object deserialization.
   */
  def loadModel(saveLocation: File): Try[LdaModel] = {
    val fis = new FileInputStream(saveLocation)
    val res = loadModel(fis)
    fis.close()
    res
  }

  /**
   * Load a saved topic model from an input stream.
   *
   * @param modelIS input stream for model.
   * @return loaded model.
   */
  def loadModel(modelIS: InputStream): Try[LdaModel] =
    Try {
      new ObjectInputStream(modelIS)
        .readObject()
        .asInstanceOf[LdaModel]
    }

}