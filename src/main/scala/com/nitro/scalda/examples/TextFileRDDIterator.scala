package com.nitro.scalda.examples

import java.io.File

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

/** WARNING: Contains mutable state. */
class TextFileRddIterator(
    corpusDirectory: File,
    mbSize: Int,
    logging: Boolean = true
)(implicit sc: SparkContext) extends Iterator[RDD[String]] {

  // mutable state
  private[this] val fileMinibatches =
    Option(corpusDirectory.listFiles)
      .map(_.toSeq)
      .getOrElse(Seq.empty[File])
      .grouped(mbSize)

  override def hasNext =
    fileMinibatches.hasNext

  override def next() = {
    log("[RDD] next minibatch", on = logging)
    sc.parallelize {
      for {
        f <- fileMinibatches.next()
      } yield text(f)
    }
  }
}