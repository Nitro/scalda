package com.nitro.scalda.examples

import java.io.File

import com.nitro.scalda.models.OnlineLDAParams
import com.nitro.scalda.models.onlineLDA.local.LocalOnlineLDA

class TextFileIterator(corpusDirectory: String, mbSize: Int) extends Iterator[IndexedSeq[String]] {

  val directoryFile = new File(corpusDirectory)

  val fileMinibatches = directoryFile
    .listFiles()
    .filter(f => f.getName != ".DS_Store")
    .grouped(mbSize)

  def hasNext = fileMinibatches.hasNext

  def next() = {

    println("processing next minibatch...")
    val nextMb = fileMinibatches.next()
    val stringMb = nextMb.map(f => scala.io.Source.fromFile(f, "ISO-8859-1").getLines.mkString(" "))
    stringMb.toIndexedSeq

  }

}

object LocalOnlineLDAExample extends App {

  val corpusDirectory = "datasets/nips_corpus"
  val vocabFile = "datasets/nips_vocab.txt"
  val mbSize = 100
  val numTopics = 20
  val numDocs = 6000
  val myIter = new TextFileIterator(corpusDirectory, mbSize)

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
    totalDocs = numDocs,
    perplexity = true)

  val myLDA = LocalOnlineLDA(p)

  val ldaModel = myLDA.inference(myIter)

  println("<-------------TOPICS LEARNED--------------->")

  myLDA.printTopics(ldaModel)

}
