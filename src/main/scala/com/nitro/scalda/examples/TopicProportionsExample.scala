package com.nitro.scalda.examples

import com.nitro.scalda.models.onlineLDA.local.LocalOnlineLDA
import com.nitro.scalda.tokenizer.StanfordLemmatizer

import scala.io.Source

object TopicProportionsExample extends App {

  val modelLocation = args(0)
  val docLocation = args(1)

  val testDoc = Source.fromFile(docLocation).getLines().mkString

  println(testDoc)

  val lda = LocalOnlineLDA()

  val myModelTry = lda.loadModel(modelLocation)

  val myModel = myModelTry.get

  lda.printTopics(myModel)

  val lemmatizer = StanfordLemmatizer()

  val topicProps = lda.topicProportions(testDoc, myModel, Some(lemmatizer))

  println(topicProps.toList.sortBy(-_._1))

}
