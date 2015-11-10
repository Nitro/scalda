package com.nitro.scalda.lemmatizer

import java.util.Properties
import edu.stanford.nlp.ling.CoreAnnotations.{ LemmaAnnotation, TokensAnnotation, SentencesAnnotation }
import edu.stanford.nlp.ling.CoreLabel
import scala.collection.JavaConversions._
import edu.stanford.nlp.pipeline.{ Annotation, StanfordCoreNLP }
import edu.stanford.nlp.util.CoreMap

object StanfordLemmatizer {

  val props = new Properties()

  props.put("annotators", "tokenize, ssplit, pos, lemma")

  val pipeline = new StanfordCoreNLP(props)

  def apply() = {
    new StanfordLemmatizer(pipeline)
  }
}

class StanfordLemmatizer(pipeline: StanfordCoreNLP) extends Lemmatizer[String] {

  override def lemmatize(rawText: String): Seq[String] = {

    val doc = new Annotation(rawText)

    pipeline.annotate(doc)

    val sentences = doc.get[java.util.List[CoreMap], SentencesAnnotation](classOf[SentencesAnnotation])

    sentences.flatMap { s =>
      val tokens = s.get[java.util.List[CoreLabel], TokensAnnotation](classOf[TokensAnnotation])
      tokens.map(_.get[String, LemmaAnnotation](classOf[LemmaAnnotation]))
    }

  }

}
