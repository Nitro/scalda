package com.nitro.scalda.examples

import java.io.File

import com.nitro.scalda.models.OnlineLDAParams
import com.nitro.scalda.models.onlineLDA.distributed.DistributedOnlineLDA
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.rdd.RDD

class textFileRDDIterator(corpusDirectory: String, mbSize: Int)(implicit sc: SparkContext) extends Iterator[RDD[String]] {

  val directoryFile = new File(corpusDirectory)

  val fileMinibatches = directoryFile
    .listFiles()
    .grouped(mbSize)

  def hasNext = fileMinibatches.hasNext

  def next() = {

    val nextMb = fileMinibatches.next()
    val stringMb = nextMb.map(f => scala.io.Source.fromFile(f, "ISO-8859-1").getLines.mkString)

    sc.parallelize(stringMb)
  }

}

object DistributedOnlineLDAExample extends App {

  val corpusDirectory = args(0)
  val vocabFile = args(1)
  val mbSize = args(2).toInt
  val numTopics = args(3).toInt
  val numDocs = args(4).toInt

  val conf = new SparkConf()
    .setAppName("Distributed Online LDA Example")
    .setMaster("local[3]")

  implicit val sc = new SparkContext(conf)

  val myIter = new textFileRDDIterator(corpusDirectory, mbSize)

  val vocab = scala.io.Source.fromFile(vocabFile).getLines.toList

  val p = OnlineLDAParams(
    vocabulary = vocab,
    alpha = 1.0 / numTopics,
    eta = 1.0 / numTopics,
    decay = 1024,
    learningRate = 0.7,
    maxIter = 100,
    convergenceThreshold = 0.001,
    numTopics = numTopics,
    totalDocs = numDocs)

  val ldaModel = DistributedOnlineLDA.inference(myIter, p)

  DistributedOnlineLDA.printTopics(ldaModel)

  sc.stop()
}
