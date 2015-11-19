package com.nitro.scalda.models.onlineLDA.distributed

import breeze.linalg.DenseMatrix
import breeze.numerics._
import breeze.stats.distributions.Gamma
import breeze.stats.mean
import com.nitro.scalda.Utils
import com.nitro.scalda.models._
import com.nitro.scalda.tokenizer.StanfordLemmatizer
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{ IndexedRow, IndexedRowMatrix }
import org.apache.spark.rdd.RDD

object DistributedOnlineLDA extends OnlineLDA {

  override type MinibatchSStats = MbSStats[Array[(Int, Array[Double])], Array[Double]]
  override type SStats = ModelSStats[IndexedRowMatrix]
  override type TopicMatrix = Array[Array[Double]]

  type MatrixRow = Array[Double]

  /**
   * Perform the E-Step on one document (to be performed in parallel via a map)
   * @param doc document from corpus.
   * @param currentTopics topics that have been learned so far
   * @param params Online LDA parameters.
   * @return Sufficient statistics for this minibatch.
   */
  def oneDocEStep(doc: Document,
                  currentTopics: TopicMatrix,
                  params: ModelParams): MinibatchSStats = {

    val wordIds = doc.wordIds
    val wordCts = DenseMatrix(doc.wordCts.map(_.toDouble))

    var gammaDoc = new DenseMatrix[Double](1,
      params.numTopics,
      Gamma(100.0, 1.0 / 100.0).sample(params.numTopics).toArray)

    var expELogThetaDoc = exp(Utils.dirichletExpectation(gammaDoc))
    val currentTopicsMatrix = new DenseMatrix(currentTopics.size,
      params.numTopics,
      currentTopics.flatten,
      0,
      params.numTopics,
      true)
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
   * Merge the rows of the overall topic matrix and the minibatch topic matrix
   * @param lambdaRow row from overall topic matrix
   * @param updateRow row from minibatch topic matrix
   * @param numUpdates total number of updates
   * @param mbSize number of documents in the minibatch
   * @param params onlineLDA parameters
   * @return merged row.
   */
  def mStep(lambdaRow: MatrixRow,
            updateRow: MatrixRow,
            numUpdates: Int,
            mbSize: Double,
            params: ModelParams): MatrixRow = {

    val rho = math.pow(params.decay + numUpdates, -params.learningRate)

    val updatedLambda1 = lambdaRow.map(_ * (1 - rho))
    val updatedLambda2 = updateRow.map(_ * (params.totalDocs.toDouble / mbSize) * rho + params.eta)

    Utils.arraySum(updatedLambda1, updatedLambda2)
  }

  /**
   * Learn a topic model in parallel by processing each minibatch in parallel
   * @param mbIt Iterator over RDDs of minibatches
   * @param params online LDA parameters
   * @param sc spark context
   * @return learned online LDA model.
   */
  def inference(mbIt: Iterator[RDD[String]],
                params: OnlineLDAParams)(implicit sc: SparkContext): SStats = {

    val vocabMapping: Map[String, Int] = params
      .vocabulary
      .distinct
      .zipWithIndex
      .toMap

    val lambdaDriver = new DenseMatrix[Double](
      vocabMapping.size,
      params.numTopics,
      Gamma(100.0, 1.0 / 100.0).sample(vocabMapping.size * params.numTopics).toArray)

    var lambda = new IndexedRowMatrix(
      sc.parallelize(
        Utils.denseMatrix2IndexedRows(lambdaDriver)
      )
    )


    val lemmatizer = params.lemmatize match {
      case true => Some(StanfordLemmatizer())
      case _    => None
    }

    var numUpdates = 0

    while (mbIt.hasNext) {

      numUpdates += 1

      if (numUpdates % 5 == 0) printTopics(ModelSStats[IndexedRowMatrix](lambda, vocabMapping, numUpdates))

      //get next minibatch RDD and convert to bag-of-words form
      val rawMinibatch = mbIt.next()
      val bowMinibatch = rawMinibatch.map(doc => Utils.toBagOfWords(doc, vocabMapping, lemmatizer))
      val mbSize = rawMinibatch.count().toInt

      //flatMap minibatch to RDD of (wordId, (wordCount, docId)) tuples such that we can later join by wordId
      val wordIdDocIdCount = bowMinibatch
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
        .groupByKey()
        .map { case (docId, wdIdCtRow) => wdIdCtRow.toArray }

      //Perform e-step on documents as a map, then reduce results by wordId.
      val eStepResult = wordIdCountRow
        .map(wIdCtRow => {
          oneDocEStep(
            Document(wIdCtRow.map(_._1), wIdCtRow.map(_._2)), wIdCtRow.map(_._3.toArray), params
          )
        })
        .flatMap(updates => updates.topicUpdates)
        .reduceByKey(Utils.arraySum)

      //Perform m-step by joining with lambda and merging corresponding rows by weighted sum
      val newLambdaRows = lambda.rows.map(r => (r.index.toInt, r.vector.toArray))
        .leftOuterJoin(eStepResult)
        .map {
          case (rowID, (lambdaRow, rowUpdate)) =>
            IndexedRow(
              rowID,
              Vectors.dense(mStep(lambdaRow, rowUpdate.getOrElse(Array.fill(params.numTopics)(0.0)), numUpdates, mbSize, params)))
        }

      lambda = new IndexedRowMatrix(newLambdaRows)
    }

    ModelSStats[IndexedRowMatrix](lambda, vocabMapping, numUpdates)
  }

  /**
   * Print the top 10 words by probability for each topic from a learned topic model
   * @param model A learned topic model.
   */
  def printTopics(model: SStats): Unit = {

    val reverseVocab = model
      .vocabMapping
      .map(_.swap)

    val rowRDD = model.lambda
      .rows
      .map(x => x.vector.toArray.zipWithIndex.map(y => (y._2, (x.index, y._1))))
      .flatMap(z => z)

    val columns = rowRDD.groupByKey()
    val sortedColumns = columns.sortByKey(true)

    val normalizedSortedColumns = sortedColumns
      .map(x => (x._1, x._2, x._2.toList.map(y => y._2).sum))
      .map(x => (x._1, x._2.toList.map(y => (y._1, y._2 / x._3))))

    val top = normalizedSortedColumns
      .map(t => t._2.sortBy(-_._2).take(10))
      .collect()

    val topN = top
      .map(x => x.map(y => (reverseVocab(y._1.toInt), y._2)))

    for (topic <- topN.zipWithIndex) {
      println("Topic #"+topic._2+": "+topic._1)
    }

  }

  def convertToLocal(model: SStats): ModelSStats[DenseMatrix[Double]] = {

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
