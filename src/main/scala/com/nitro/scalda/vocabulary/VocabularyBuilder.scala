package com.nitro.scalda.vocabulary

import java.io.File
import com.nitro.scalda.lemmatizer.StanfordLemmatizer
import com.nitro.scalda.tokenizer.StanfordTokenizer

import scala.io.Source
import scala.util.matching.Regex

case class WordDocCount(wordCount: Int, docFrequency: Int)

object VocabularyBuilder {

  def apply(minWordCount: Int, minDocFreq: Int) = {
    new VocabularyBuilder(minWordCount, minDocFreq)
  }

}

/**
 * Pretty much a one-document-at-a-time word count so that it is scalable in terms of memory.
 * @param minWordCount minimum total word count for a word to be part of the vocabulary
 * @param minDocFreq minimum document frequency for a word to be part of the vocabulary
 */
class VocabularyBuilder(minWordCount: Int, minDocFreq: Int) {

  val vocabTokenizer = StanfordTokenizer
  val vocabLemmatizer = StanfordLemmatizer()
  val stopWords = Source.fromInputStream(getClass.getResourceAsStream("/stop-word-list.txt")).getLines().toSet
  val lettersPattern = new Regex("[A-Za-z]")

  def textFileVocab(corpusDirectory: String): Seq[String] = {

    var wordStats: Map[String, WordDocCount] = Map.empty

    val docFiles = new File(corpusDirectory)
      .listFiles()
      .filter(f => f.getName != ".DS_Store")

    var docsProcessed = 0

    println("beginning word count")
    //Traverse document files, read => tokenize => bag-of-words => update WordDocCount.
    docFiles.foreach { doc =>

      if (docsProcessed % 100 == 0) println(s"${docsProcessed} documents counted")

      val content = Source.fromFile(doc, "ISO-8859-1")
        .getLines()
        .mkString(" ")

      val bow = vocabTokenizer
        .tokenize(content)
        .filter(w => !stopWords.contains(w) && lettersPattern.findFirstIn(w).isDefined && w.length > 2)
        .groupBy(identity)
        .mapValues(_.size)

      bow.foreach { wFreq =>

        wordStats.get(wFreq._1) match {

          case Some(wdc) => {

            wordStats += (wFreq._1 -> wordStats(wFreq._1)
              .copy(wordCount = wdc.wordCount + wFreq._2, docFrequency = wdc.docFrequency + 1))
          }

          case _ => wordStats += (wFreq._1 -> WordDocCount(wFreq._2, 1))
        }

      }

      docsProcessed += 1

    }

    wordStats.keys.foreach(println)

    //map tokens to their lemmatized versions.  Sum token word counts/doc frequencies for tokens that map to the same lemma
    var lemmaStats = Map.empty[String, WordDocCount]

    println("beginning vocab lemmatization")

    var lemmaCounter = 0
    wordStats.keys.foreach { token =>

      if (lemmaCounter % 100 == 0) println(s"${lemmaCounter} vocab words lemmatized")

      val lemma = vocabLemmatizer.lemmatize(token)

      lemmaStats.get(lemma.head) match {

        case Some(ldc) => lemmaStats += (lemma.head -> lemmaStats(lemma.head)
          .copy(
            wordCount = lemmaStats(lemma.head).wordCount + wordStats(token).wordCount,
            docFrequency = lemmaStats(lemma.head).docFrequency + wordStats(token).docFrequency
          ))

        case _ => lemmaStats += (lemma.head -> wordStats(token))
      }

      lemmaCounter += 1

    }

    //filter according to specs
    lemmaStats
      .filter(w => (w._2.wordCount >= minWordCount) && (w._2.docFrequency >= minDocFreq) && !stopWords.contains(w._1) && w._1.length > 2)
      .keys
      .toSeq

  }

}
