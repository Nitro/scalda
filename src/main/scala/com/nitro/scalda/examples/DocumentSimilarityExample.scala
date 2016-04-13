package com.nitro.scalda.examples

import java.io.File

import com.nitro.scalda.models.OnlineLdaParams
import com.nitro.scalda.models.onlineLDA.local.LocalOnlineLda

object DocumentSimilarityExample extends App {

  val doc1Location = new File(args(0))
  val doc2Location = new File(args(1))
  val modelLocation = new File(args(2))
  val vocabFile = new File(args(3))
  val numTopics = args(4).toInt
  val numDocs = args(5).toInt

  log(
    s"""[DocumentSimilarityExample]
       |Document 1 location:      $doc1Location
       |Document 2 location:      $doc2Location
       |Saved LDA model location: $modelLocation
       |Vocabulary file:          $vocabFile
       |Number of topics:         $numTopics
       |Corpus size:              $numDocs
       |--------------------------
     """.stripMargin
  )

  val lda = LocalOnlineLda(
    OnlineLdaParams(
      lines(vocabFile).toIndexedSeq,
      1.0 / numTopics,
      1.0 / numTopics,
      1024,
      0.7,
      100,
      0.001,
      numTopics,
      numDocs
    )
  )

  val model = lda.loadModel(modelLocation).get

  val docSim12 = lda.documentSimilarity(
    text(doc1Location),
    text(doc2Location),
    model
  )
  println("<---------- DOCUMENT SIMILARITY ----------->")
  println(docSim12)
}