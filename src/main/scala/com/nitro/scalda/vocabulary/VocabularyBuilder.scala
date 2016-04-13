package com.nitro.scalda.vocabulary

import java.io.File
import com.nitro.scalda.tokenizer.{ StanfordLemmatizer, StanfordTokenizer }
import scala.io.Source
import scala.util.matching.Regex

case class WordDocCount(wordCount: Int, docFrequency: Int)

object VocabularyBuilder {

  def apply(
    minWordCount: Int,
    minDocFreq: Int,
    lem: Boolean = false
  ) = {
    new VocabularyBuilder(minWordCount, minDocFreq, lem)
  }

}

/**
 * Pretty much a one-document-at-a-time word count so that it is scalable in terms of memory.
 * @param minWordCount minimum total word count for a word to be part of the vocabulary
 * @param minDocFreq minimum document frequency for a word to be part of the vocabulary
 */
class VocabularyBuilder(minWordCount: Int, minDocFreq: Int, lem: Boolean) {

  //If lemmatizer is enabled, tokens are also lemmatized
  val tokenizer = lem match {
    case true => StanfordLemmatizer()
    case _ => StanfordTokenizer
  }

  //Load set of stop words
  val stopWords = Source
    .fromFile("src/main/resources/stop-word-list.txt")
    .getLines()
    .toSet

  //regex that matches on tokens containing at least one letter
  val lettersPattern = new Regex("[A-Za-z]")

  //create a vocabulary from text files located in some local directory
  def buildTextFileVocab(corpusDirectory: File): Seq[String] = {

    //list files
    val docFiles =
      Option(corpusDirectory.listFiles)
        .map(_.toSeq)
        .getOrElse(Seq.empty[File])
        .filter { f => !f.getName.startsWith(".") }

    //fold over files, reading them and updating word counts and doc frequencies
    val wordStats = docFiles
      .foldLeft(collection.mutable.Map.empty[String, WordDocCount]) {
        (acc, docFile) =>

          //read file
          val content = Source.fromFile(docFile, "ISO-8859-1")
            .getLines()
            .mkString(" ")

          //transform to bag-of-words form via tokenization/lemmatization.
          // Also remove short and non-alphanumeric words.
          val bagOfWords = tokenizer
            .tokenize(content)
            .filter(w => !stopWords.contains(w) && lettersPattern.findFirstIn(w).isDefined && w.length > 2)
            .groupBy(identity)
            .mapValues(_.size)

          //update accumulator counts
          bagOfWords.foreach { wordFreq =>
            acc.get(wordFreq._1) match {
              case Some(wdc) =>
                acc += (wordFreq._1 -> wdc.copy(
                  wordCount = wdc.wordCount + wordFreq._2,
                  docFrequency = wdc.docFrequency + 1
                ))
              case _ =>
                acc += (wordFreq._1 -> WordDocCount(wordFreq._2, 1))
            }
          }

          acc
      }

    //filter according to minWordCount, minDocFreq.
    wordStats
      .filter(w => w._2.wordCount >= minWordCount && w._2.docFrequency >= minDocFreq)
      .keys
      .toSeq
  }

}
