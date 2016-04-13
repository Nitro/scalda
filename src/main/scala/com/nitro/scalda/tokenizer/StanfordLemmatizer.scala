package com.nitro.scalda.tokenizer

import java.util.Properties

import edu.stanford.nlp.ling.CoreAnnotations.{ LemmaAnnotation, SentencesAnnotation, TokensAnnotation }
import edu.stanford.nlp.ling.CoreLabel
import edu.stanford.nlp.pipeline.{ Annotation, StanfordCoreNLP }
import edu.stanford.nlp.util.CoreMap

import scala.collection.JavaConversions._

object StanfordLemmatizer {

  val props = new Properties()

  props.put("annotators", "tokenize, ssplit, pos, lemma")

  val pipeline = new StanfordCoreNLP(props)

  def apply() = {
    new StanfordLemmatizer(pipeline)
  }
}

//Let's assume that a lemmatizer is just a special case of a tokenizer when the tokens are lemmatized.
class StanfordLemmatizer(pipeline: StanfordCoreNLP) extends Tokenizer[String] {

  override def tokenize(rawText: String): Seq[String] = {

    val doc = new Annotation(rawText)

    pipeline.annotate(doc)

    val sentences = doc.get[java.util.List[CoreMap], SentencesAnnotation](classOf[SentencesAnnotation])

    sentences.flatMap { s =>
      val tokens = s.get[java.util.List[CoreLabel], TokensAnnotation](classOf[TokensAnnotation])
      tokens.map(_.get[String, LemmaAnnotation](classOf[LemmaAnnotation]))
    }

  }

}
