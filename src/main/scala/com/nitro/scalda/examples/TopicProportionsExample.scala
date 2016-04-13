package com.nitro.scalda.examples

import java.io.File

import com.nitro.scalda.models.onlineLDA.local.LocalOnlineLda
import com.nitro.scalda.tokenizer.StanfordLemmatizer

object TopicProportionsExample extends App {

  val modelLocation = new File(args(0))
  val docLocation = new File(args(1))

  log(
    s"""[TopicProportionsExample]
       |Saved LDA model location: $modelLocation
       |Single document location: $docLocation
       |--------------------------
     """.stripMargin
  )

  val testDoc = text(docLocation)

  log(s"Document text:\n$testDoc\n")

  val lda = LocalOnlineLda.empty

  val model = lda.loadModel(modelLocation).get

  println("<------------ TOPICS LEARNED -------------->")
  lda.printTopics(model)

  val topicProps = lda.topicProportions(
    testDoc,
    model,
    Some(StanfordLemmatizer())
  )

  println("<----------- TOPIC PROPORTIONS ------------>")
  println(
    topicProps
      .toSeq
      .sortBy { case (prob, _) => -prob }
      .map {
        case (prob, topicIdx) => s"$prob,$topicIdx"
      }
      .mkString("\n")
  )
}