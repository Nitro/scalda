package com.nitro.scalda.examples

import com.nitro.scalda.vocabulary.VocabularyBuilder

object CreateVocabulary extends App {

  //User must specify the location of the directory containing the text file documents
  val corpusDirectory = args(0)

  //Create vocab containing words with minimum word count of 5 and minimum document frequency of 5
  val myVocab = VocabularyBuilder(
    minWordCount = 5,
    minDocFreq = 5
  )
    .buildTextFileVocab(corpusDirectory)

  //print each word in vocab
  myVocab.foreach(println)

  //print length of vocab
  println(myVocab.length)
}
