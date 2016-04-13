package com.nitro.scalda.tokenizer

import java.io.{ StringReader, BufferedReader }

import edu.stanford.nlp.process.PTBTokenizer
import scala.collection.JavaConversions._

object StanfordTokenizer extends Tokenizer[String] {

  def tokenize(text: String): Seq[String] = {

    val tokenizer = PTBTokenizer.newPTBTokenizer(new BufferedReader(new StringReader(text)))

    tokenizer
      .tokenize()
      .map(_.word().toLowerCase())
      .toSeq
  }

}
