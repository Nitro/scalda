package com.nitro.scalda.examples

import com.nitro.scalda.models.OnlineLDAParams
import com.nitro.scalda.models.onlineLDA.local.LocalOnlineLDA
import com.nitro.scalda.tokenizer.StanfordLemmatizer

import scala.io.Source

object DocumentSimilarityExample extends App {

  val doc1Location = args(0)
  val doc2Location = args(1)
  val modelLocation = args(2)
  val vocabFile = args(3)
  val mbSize = args(4).toInt
  val numTopics = args(5).toInt
  val numDocs = args(6).toInt

  val doc1 = Source.fromFile(doc1Location).getLines().mkString
  val doc2 = Source.fromFile(doc2Location).getLines().mkString

  val vocab = scala.io.Source.fromFile(vocabFile).getLines.toList

  val p = OnlineLDAParams(
    vocab,
    1.0 / numTopics,
    1.0 / numTopics,
    1024,
    0.7,
    100,
    0.001,
    numTopics,
    numDocs)

  val lda = LocalOnlineLDA(p)

  val myModel = lda.loadModel(modelLocation)

  val lemmatizer = StanfordLemmatizer()

  val docSim12 = lda.documentSimilarity(doc1, doc2, myModel.get)

}
