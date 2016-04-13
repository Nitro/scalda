package com.nitro.scalda.models.onlineLDA.distributed

import breeze.linalg.DenseMatrix
import breeze.linalg.sum
import breeze.numerics._
import breeze.stats.distributions.Gamma
import breeze.stats.mean
import com.nitro.scalda.Utils
import com.nitro.scalda.models._
import com.nitro.scalda.tokenizer.StanfordLemmatizer
import com.nitro.scalda.evaluation.perplexity.Perplexity._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{ DenseVector, Vectors }
import org.apache.spark.mllib.linalg.distributed.{ IndexedRow, IndexedRowMatrix }
import org.apache.spark.rdd.RDD

class DistributedOnlineLda(params: OnlineLdaParams) extends OnlineLda with Serializable {

  override type BowMinibatch = RDD[Document]
  override type MinibatchSStats = (RDD[(Int, Array[Double])], Int)
  override type LdaModel = ModelSStats[IndexedRowMatrix]
  override type Lambda = IndexedRowMatrix
  override type Minibatch = RDD[String]

  type MatrixRow = Array[Double]

  /**
   * Perform E-Step on minibatch.
   *
   * @param mb
   * @param lambda
   * @param gamma
   * @return The sufficient statistics for this minibatch
   */
  override def eStep(
    mb: BowMinibatch,
    lambda: Lambda,
    gamma: Gamma
  ): MinibatchSStats = {

    val mbSize = mb.count().toInt

    val wordIdDocIdCount = mb
      .zipWithIndex()
      .flatMap {
        case (docBOW, docId) =>
          docBOW.wordIds.zip(docBOW.wordCts)
            .map { case (wordId, wordCount) => (wordId, (wordCount, docId)) }
      }

    //Join with lambda to get RDD of (wordId, ((wordCount, docId), lambda(wordId),::)) tuples
    val wordIdDocIdCountRow = wordIdDocIdCount
      .join(lambda.rows.map(row => (row.index.toInt, row.vector.toArray)))

    //Now group by docID in order to recover documents
    val wordIdCountRow = wordIdDocIdCountRow
      .map { case (wordId, ((wordCount, docId), wordRow)) => (docId, (wordId, wordCount, wordRow)) }

    val removedDocId = wordIdCountRow
      .groupByKey()
      .map { case (docId, wdIdCtRow) => wdIdCtRow.toArray }

    //Perform e-step on documents as a map, then reduce results by wordId.
    val eStepResult = removedDocId
      .map(wIdCtRow => {
        oneDocEStep(
          Document(wIdCtRow.map(_._1), wIdCtRow.map(_._2)),
          gamma,
          wIdCtRow.map(_._3.toArray)
        )
      })

    //collect RDDs and and compute perplexity in driver.  At some point this should be "sparkified" for speed.
    if (params.perplexity) {

      val usedDocs = wordIdCountRow
        .map(_._1.toInt)
        .collect()
        .toSet

      val localMb = mb.zipWithIndex()
        .filter { case (doc, id) => usedDocs.contains(id.toInt) }
        .map(_._1)
        .collect()

      val localLambda = Utils.rdd2DM(lambda.rows)

      val gammaRDD = eStepResult
        .zipWithIndex()
        .map { case (x, idx) => IndexedRow(idx, new DenseVector(x.topicProportions)) }

      val localGamma = Utils.rdd2DM(gammaRDD)

      val score = perplexity(localMb, localGamma, localLambda.t, params)

      val mbWordCt = sum(localMb.flatMap(_.wordCts))
      val wordScale = (score * localMb.size) / (params.totalDocs * mbWordCt.toDouble)

      println(s"per-word perplexity: ${exp(-wordScale)}")
    }

    val topicUpdates = eStepResult
      .flatMap(updates => updates.topicUpdates)
      .reduceByKey(Utils.arraySum)

    (topicUpdates, mbSize)
  }

  /**
   * Perform the E-Step on one document (to be performed in parallel via a map)
   *
   * @param doc document from corpus.
   * @param currentTopics topics that have been learned so far
   * @return Sufficient statistics for this minibatch.
   */
  def oneDocEStep(
    doc: Document,
    gamma: DenseMatrix[Double],
    currentTopics: Array[Array[Double]]
  ): MbSStats[Array[(Int, Array[Double])], Array[Double]] = {

    val wordIds = doc.wordIds
    val wordCts = DenseMatrix(doc.wordCts.map(_.toDouble))

    var gammaDoc = gamma

    var expELogThetaDoc = exp(Utils.dirichletExpectation(gammaDoc))
    val currentTopicsMatrix = new DenseMatrix(
      currentTopics.size,
      params.numTopics,
      currentTopics.flatten,
      0,
      params.numTopics,
      true
    )
    val expELogBetaDoc = exp(Utils.dirichletExpectation(currentTopicsMatrix.t))

    var phiNorm = expELogThetaDoc * expELogBetaDoc + 1e-100

    var convergence = false
    var iter = 0

    //begin update iteration.  Stop if convergence has occurred or we reach the max number of iterations.
    while (iter < params.maxIter && !convergence) {

      val lastGammaD = gammaDoc.t
      val gammaPreComp = expELogThetaDoc :* (wordCts / phiNorm.t) * expELogBetaDoc.t
      gammaDoc = gammaPreComp + params.alpha
      expELogThetaDoc = exp(Utils.dirichletExpectation(gammaDoc))
      phiNorm = expELogThetaDoc * expELogBetaDoc + 1e-100

      if (mean(abs(gammaDoc.t - lastGammaD)) < params.convergenceThreshold) convergence = true

      iter += 1
    }

    val lambdaUpdatePreCompute: DenseMatrix[Double] = expELogThetaDoc.t * (wordCts / phiNorm)

    //Compute lambda row updates and zip with rowIds (note: rowIds = wordIds)
    val lambdaUpdate = (lambdaUpdatePreCompute :* expELogBetaDoc)
      .toArray
      .grouped(lambdaUpdatePreCompute.rows)
      .toArray
      .zip(wordIds)
      .map(x => (x._2, x._1))

    MbSStats(lambdaUpdate, gammaDoc.toArray)
  }

  /**
   * Perform m-step by updating the current model with the minibatch sufficient statistics.
   *
   * @param model
   * @param mSStats
   * @return Updated LDA model
   */
  override def mStep(
    model: LdaModel,
    mSStats: MinibatchSStats
  ): LdaModel = {

    val mbSize = mSStats._2
    val topicUpdates = mSStats._1

    val newLambdaRows = model
      .lambda
      .rows
      .map(r => (r.index.toInt, r.vector.toArray))
      .leftOuterJoin(topicUpdates)
      .map {
        case (rowID, (lambdaRow, rowUpdate)) =>
          IndexedRow(
            rowID,
            Vectors.dense(
              oneDocMStep(
                lambdaRow,
                rowUpdate.getOrElse(Array.fill(params.numTopics)(0.0)),
                model.numUpdates,
                mbSize
              )
            )
          )
      }

    val newTopics = new IndexedRowMatrix(newLambdaRows)

    model.copy(lambda = newTopics)
  }

  /**
   * Merge the rows of the overall topic matrix and the minibatch topic matrix
   *
   * @param lambdaRow row from overall topic matrix
   * @param updateRow row from minibatch topic matrix
   * @param numUpdates total number of updates
   * @param mbSize number of documents in the minibatch
   * @return merged row.
   */
  def oneDocMStep(
    lambdaRow: MatrixRow,
    updateRow: MatrixRow,
    numUpdates: Int,
    mbSize: Double
  ): MatrixRow = {

    val rho = math.pow(params.decay + numUpdates, -params.learningRate)

    val updatedLambda1 = lambdaRow.map(_ * (1 - rho))
    val updatedLambda2 = updateRow.map(_ * (params.totalDocs.toDouble / mbSize) * rho + params.eta)

    Utils.arraySum(updatedLambda1, updatedLambda2)
  }

  /**
   * Perform inference to learn the LDA model.
   *
   * @param minibatchIterator
   * @param sc
   * @return A trained LDA model.
   */
  def inference(minibatchIterator: Iterator[RDD[String]])(implicit sc: SparkContext): LdaModel = {

    val vocabMapping: Map[String, Int] = params
      .vocabulary
      .distinct
      .zipWithIndex
      .toMap

    val lambdaDriver = new DenseMatrix[Double](
      vocabMapping.size,
      params.numTopics,
      Gamma(100.0, 1.0 / 100.0).sample(vocabMapping.size * params.numTopics).toArray
    )

    val lambda = new IndexedRowMatrix(
      sc.parallelize(
        Utils.denseMatrix2IndexedRows(lambdaDriver)
      )
    )

    var curModel = ModelSStats[IndexedRowMatrix](lambda, vocabMapping, 0)

    val lemmatizer = params.lemmatize match {
      case true => Some(StanfordLemmatizer())
      case _ => None
    }

    var mbProcessed = 0

    while (minibatchIterator.hasNext) {

      mbProcessed += 1

      if (mbProcessed % 5 == 0) printTopics(ModelSStats[IndexedRowMatrix](curModel.lambda, vocabMapping, mbProcessed))

      //get next minibatch RDD and convert to bag-of-words form
      val rawMinibatch = minibatchIterator.next()
      val bowMinibatch = rawMinibatch.map(doc => Utils.toBagOfWords(doc, vocabMapping, lemmatizer))

      val gamma = new DenseMatrix[Double](
        1,
        params.numTopics,
        Gamma(100.0, 1.0 / 100.0).sample(params.numTopics).toArray
      )

      val mbSStats = eStep(bowMinibatch, curModel.lambda, gamma)

      curModel = mStep(curModel.copy(numUpdates = mbProcessed), mbSStats)

    }

    curModel
  }

  /**
   * Print the top 10 words by probability for each topic from a learned topic model
   *
   * @param model A learned topic model.
   */
  def printTopics(model: LdaModel): Unit = {

    val reverseVocab = model
      .vocabMapping
      .map(_.swap)

    val rowRDD = model.lambda
      .rows
      .map(x => x.vector.toArray.zipWithIndex.map(y => (y._2, (x.index, y._1))))
      .flatMap(z => z)

    val columns = rowRDD.groupByKey()
    val sortedColumns = columns.sortByKey(ascending = true)

    val normalizedSortedColumns = sortedColumns
      .map(x => (x._1, x._2, x._2.toSeq.map(y => y._2).sum))
      .map(x => (x._1, x._2.toSeq.map(y => (y._1, y._2 / x._3))))

    val top = normalizedSortedColumns
      .map(t => t._2.sortBy(-_._2).take(10))
      .collect()

    val topN = top
      .map(x => x.map(y => (reverseVocab(y._1.toInt), y._2)))

    for (topic <- topN.zipWithIndex) {
      println("Topic #" + topic._2 + ": " + topic._1)
    }

  }

  /**
   * Convert a distributed online LDA model to a local online LDA model.
   *
   * @param model
   * @return
   */
  def convertToLocal(model: LdaModel): ModelSStats[DenseMatrix[Double]] = {

    val localMatrixRows = model
      .lambda
      .rows
      .collect()
      .sortBy(_.index)
      .flatMap(_.vector.toArray)

    val rows = model.lambda.numRows().toInt
    val cols = model.lambda.numCols().toInt

    ModelSStats(
      lambda = new DenseMatrix(rows, cols, localMatrixRows, 0, cols, true),
      vocabMapping = model.vocabMapping,
      numUpdates = model.numUpdates
    )
  }

}
