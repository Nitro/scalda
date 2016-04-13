package com.nitro.scalda.evaluation.coherence

import breeze.linalg.{ *, DenseMatrix }
import com.nitro.scalda.models.ModelSStats

object WordVectorCoherence {

  def cosineSimilarity(v1: Array[Float], v2: Array[Float]): Double = {

    val numerator = v1
      .zip(v2)
      .foldLeft(0.0) { (a, i) =>
        a + (i._1 * i._2)
      }

    def denominator(v: Array[Float]): Double = v
      .map(x => x * x)
      .foldLeft(0.0)(_ + _)

    numerator / (math.sqrt(denominator(v1)) * math.sqrt(denominator(v2)))
  }

  def getTopicCoherence(model: ModelSStats[DenseMatrix[Double]], wordVectors: Map[String, Array[Float]]): Array[TopicCoherence] = {

    val reverseVocab = model.vocabMapping.map(_.swap)
    val lambda = model.lambda

    //compute the average pairwise cosine similarity between word vectors corresponding to top ten words in each topic
    lambda(*, ::).map { topic =>

      val topTenWords = topic
        .toArray
        .zipWithIndex
        .sortBy(-_._1)
        .take(10)
        .map(w => reverseVocab(w._2))

      val topTenWordVectors = topTenWords.map(w => wordVectors(w))

      val allPairs = topTenWordVectors.combinations(2).toSeq

      val numPairs = allPairs.size

      val pairwiseWVCoherence = allPairs.foldLeft(0.0) { (a, i) =>
        a + cosineSimilarity(i(0), i(1))
      } / numPairs

      TopicCoherence(topTenWords.toIndexedSeq, pairwiseWVCoherence)
    }.toArray.sortBy(-_.coherence)

  }

}
