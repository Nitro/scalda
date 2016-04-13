package com.nitro.scalda

import breeze.linalg._
import breeze.numerics._
import com.nitro.scalda.models.Document
import com.nitro.scalda.tokenizer.{ StanfordLemmatizer, StanfordTokenizer }
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{ IndexedRowMatrix, IndexedRow }
import org.apache.spark.rdd.RDD

object Utils {

  def toBagOfWords(
    text: String,
    vocabMapping: Map[String, Int],
    lemmatizer: Option[StanfordLemmatizer] = None
  ): Document = {

    lazy val tokenizer = StanfordTokenizer

    val cleanText = text
      .replaceAll("[^a-zA-Z ]", "")
      .toLowerCase()

    val words = lemmatizer match {
      case Some(stanLemma) => stanLemma.tokenize(cleanText)
      case _ => tokenizer.tokenize(cleanText)
    }

    val bow = words
      .filter(w => vocabMapping.contains(w))
      .map(w => vocabMapping(w))
      .groupBy(identity)
      .mapValues(_.size)
      .toIndexedSeq

    Document(bow.map(_._1), bow.map(_._2))
  }

  def mapVocabId(vocab: Seq[String]): Map[String, Int] =
    vocab
      .distinct
      .zipWithIndex
      .toMap

  def denseMatrix2IndexedRows(dm: DenseMatrix[Double]): Array[IndexedRow] = {

    dm(*, ::)
      .map(_.toArray)
      .toArray
      .zipWithIndex
      .map { case (row, index) => IndexedRow(index, Vectors.dense(row)) }

  }

  def rdd2DM(rddRows: RDD[IndexedRow]): DenseMatrix[Double] = {

    val localRows = rddRows
      .collect()
      .sortBy(_.index)
      .map(_.vector.toArray)

    DenseMatrix(localRows: _*)
  }

  def arraySum(a1: Array[Double], a2: Array[Double]): Array[Double] = {

    a1.zip(a2).map { case (a1El1, a2El1) => a1El1 + a2El1 }

  }

  def optionArrayMultiply(a1: Array[Double], a2: Option[Array[Double]]): Array[Double] = {

    a2 match {
      case Some(update) => a1.zip(update).map(x => x._1 * x._2)
      case None => a1
    }
  }

  def dirichletExpectation(hParam: DenseMatrix[Double]): DenseMatrix[Double] =
    hParam match {

      case x if (x.rows == 1 || x.cols == 1) =>
        digamma(hParam) - digamma(sum(hParam))

      case _ =>

        val first_term = digamma(hParam)
        first_term(::, *) - digamma(sum(hParam, Axis._1))
    }

  def getUniqueWords(documents: Seq[Document]): Map[Double, Int] = {

    var uniqueWords: Map[Double, Int] = Map.empty

    documents.foreach { document =>
      document.wordIds.foreach { word =>
        if (!uniqueWords.contains(word)) {
          uniqueWords += (word.toDouble -> uniqueWords.size)
        }
      }
    }

    uniqueWords
  }

  def euclideanDistance(v1: Array[Double], v2: Array[Double]): Double = {

    val sumOfSquareDifference = v1
      .zip(v2)
      .foldLeft(0.0) { (a, i) =>

        val dif = i._1 - i._2

        a + (dif * dif)
      }

    math.sqrt(sumOfSquareDifference)
  }

  def logSumExp(x: DenseVector[Double]): Double = {
    val a = max(x)
    a + log(sum(exp(x :- a)))
  }

}
