package com.nitro.scalda.examples

import java.io.PrintWriter

import com.nitro.scalda.vocabulary.VocabularyBuilder

object CreateVocabulary extends App {

  val corpusDirectory = args(0)
  val vocabWriteLocation = args(1)

  val pr = new PrintWriter(vocabWriteLocation)

  val myVocab = VocabularyBuilder(50, 50).textFileVocab(corpusDirectory)

  myVocab.foreach(println)

  myVocab.foreach { w =>

    pr.println(w)
  }

  pr.close()

}
