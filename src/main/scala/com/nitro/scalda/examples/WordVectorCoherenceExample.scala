package com.nitro.scalda.examples

import java.io.File

import com.nitro.scalda.evaluation.coherence.{ WordVectorCoherence, SparkWord2Vec }
import com.nitro.scalda.models.onlineLDA.local.LocalOnlineLDA
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.rdd.RDD

import scala.io.Source

object WordVectorCoherenceExample extends App {

  val modelLocation = args(0)
  val corpusLocation = args(1)

  val conf = new SparkConf()
    .setAppName("Distributed Online LDA Example")
    .setMaster("local[3]")

  implicit val sc = new SparkContext(conf)

  def textFiles2RDD(filesDirectory: String)(implicit sc: SparkContext): RDD[String] = {

    val files = new File(filesDirectory).listFiles()

    val filesRDD = sc.parallelize(files)

    filesRDD.map { f =>
      Source.fromFile(f, "ISO-8859-1").getLines().mkString(" ")
    }

  }

  val lda = LocalOnlineLDA()

  lazy val myModel = lda.loadModel(modelLocation)

  val docRDD = textFiles2RDD(corpusLocation)

  val wordVectors = SparkWord2Vec.learnWordVectors(docRDD)

  val topicCoherence = WordVectorCoherence.getTopicCoherence(myModel.get, wordVectors)

  topicCoherence.foreach(println)

}
