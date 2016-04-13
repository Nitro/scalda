package com.nitro.scalda.examples

import java.io.File

import com.nitro.scalda.models.OnlineLdaParams
import com.nitro.scalda.models.onlineLDA.local.LocalOnlineLda

object LocalOnlineLdaExample extends App {

  val getArg = getOrElse(args) _

  val corpusDirectory = new File(getArg(0, "datasets/nips_corpus"))
  val vocabFile = new File(getArg(1, "datasets/nips_vocab.txt"))
  val mbSize = getOrElse(args, 2, 100, _.toInt)
  val numTopics = getOrElse(args, 3, 20, _.toInt)
  val numDocs = getOrElse(args, 4, 6000, _.toInt)

  log(
    s"""[LocalOnlineLdaExample]
       |Corpus directory:  $corpusDirectory
       |Vocabulary file:   $vocabFile
       |Minibatch size:    $mbSize
       |Number of topics:  $numTopics
       |Corpus size:       $numDocs
       |-------------------
     """.stripMargin
  )

  val lda = LocalOnlineLda(
    OnlineLdaParams(
      vocabulary = lines(vocabFile).toIndexedSeq,
      alpha = 1.0 / numTopics,
      eta = 1.0 / numTopics,
      decay = 1024,
      learningRate = 0.7,
      maxIter = 100,
      convergenceThreshold = 0.001,
      numTopics = numTopics,
      totalDocs = numDocs,
      perplexity = true
    )
  )

  val model = lda.inference(new TextFileIterator(corpusDirectory, mbSize))

  println("<-------------TOPICS LEARNED--------------->")
  lda.printTopics(model)
}