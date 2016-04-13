package com.nitro.scalda.examples

import java.io.File

import com.nitro.scalda.evaluation.coherence.{ WordVectorCoherence, SparkWord2Vec }
import com.nitro.scalda.models.onlineLDA.local.LocalOnlineLda
import org.apache.spark.{ SparkConf, SparkContext }

object WordVectorCoherenceExample extends App {

  val modelLoc = new File(args(0))
  val corpusLoc = new File(args(1))

  log(
    s"""[WordVectorCoherenceExample]
       |Previously saved LDA model location: $modelLoc
       |Text file corpus directory:          $corpusLoc
       |-------------------------------------
     """.stripMargin
  )

  val model = LocalOnlineLda.empty.loadModel(modelLoc).get

  implicit val sc = new SparkContext(
    new SparkConf()
      .setAppName("Word Vector Coherence Example")
      .setMaster("local[3]")
  )

  val docRDD = sc
    .parallelize {
      Option(corpusLoc.listFiles())
        .map { _.toSeq }
        .getOrElse(Seq.empty[File])
    }
    .map { text }

  val wordVecs = SparkWord2Vec.learnWordVectors(docRDD)

  val topicCoherence = WordVectorCoherence.getTopicCoherence(model, wordVecs)

  println("<------------ TOPIC COHERENCE ------------->")
  topicCoherence.foreach { println }

  sc.stop()
}