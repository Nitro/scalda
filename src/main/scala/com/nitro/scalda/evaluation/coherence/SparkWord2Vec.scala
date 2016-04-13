package com.nitro.scalda.evaluation.coherence

import java.io._

import com.nitro.scalda.tokenizer.StanfordTokenizer
import org.apache.spark.mllib.feature.Word2Vec
import org.apache.spark.rdd.RDD

object SparkWord2Vec {

  val tokenizer = StanfordTokenizer

  def learnWordVectors(docs: RDD[String]): Map[String, Array[Float]] = {

    val w2vInput = docs.map(doc => tokenizer.tokenize(doc))
    val w2v = new Word2Vec()
    val trainedModel = w2v.fit(w2vInput)

    trainedModel.getVectors
  }

  def saveWordVectors(wv: Map[String, Array[Float]], saveLocation: String): Unit = {

    val fos = new FileOutputStream(new File(saveLocation))
    val oos = new ObjectOutputStream(fos)

    oos.writeObject(wv)
    oos.close()
  }

  def loadWordVectors(saveLocation: String): Map[String, Array[Float]] = {

    val fis = new FileInputStream(new File(saveLocation))
    val ois = new ObjectInputStream(fis)

    ois.readObject().asInstanceOf[Map[String, Array[Float]]]
  }
}
