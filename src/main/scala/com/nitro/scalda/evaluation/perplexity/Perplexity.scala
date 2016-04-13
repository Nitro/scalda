package com.nitro.scalda.evaluation.perplexity

import breeze.linalg.Axis
import breeze.numerics._
import com.nitro.scalda.Utils
import breeze.linalg.DenseMatrix
import breeze.linalg.sum
import com.nitro.scalda.models.{ OnlineLdaParams, Document }

object Perplexity {

  /**
   * Minibatch perplexity calculation
   *
   * @param mb
   * @param mbGamma
   * @param lambda
   * @param params
   * @return Perplexity value for the given minibatch
   */
  def perplexity(
    mb: Seq[Document],
    mbGamma: DenseMatrix[Double],
    lambda: DenseMatrix[Double],
    params: OnlineLdaParams
  ): Double = {

    val eLogTheta = Utils.dirichletExpectation(mbGamma)
    val eLogBeta = Utils.dirichletExpectation(lambda)

    var perplexityScore = 0.0

    for ((doc, docId) <- mb.zipWithIndex) {

      val eLogThetaDoc = eLogTheta(docId, ::).t

      perplexityScore += sum(
        doc.wordIds.zip(doc.wordCts).map {

          case (wordId, wordCt) => Utils.logSumExp(eLogThetaDoc + eLogBeta(::, wordId)) * wordCt.toDouble
        }
      )

    }

    perplexityScore += sum(mbGamma.map(el => params.alpha - el) :* eLogTheta)
    perplexityScore += sum(lgamma(mbGamma) - lgamma(params.alpha))
    perplexityScore += sum(lgamma(params.alpha * params.numTopics) - lgamma(sum(mbGamma, Axis._1)))
    perplexityScore *= params.totalDocs / mb.size.toDouble
    perplexityScore += sum(lambda.map(el => params.eta - el) :* eLogBeta)
    perplexityScore += sum(lgamma(lambda) - lgamma(params.eta))
    perplexityScore += sum(lgamma(params.eta * params.vocabulary.size) - lgamma(sum(lambda, Axis._1)))

    perplexityScore
  }

}
