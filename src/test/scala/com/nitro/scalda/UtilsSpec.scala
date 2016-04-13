package com.nitro.scalda

import com.nitro.scalda.models.Document
import com.nitro.scalda.tokenizer.StanfordLemmatizer
import org.scalatest.{ Matchers, WordSpec }

class UtilsSpec extends WordSpec with Matchers {

  "Utils" should {

    "compute the correct bag-of-words" in {

      val doc = "I went to the store.  Then I went to the movies."

      val vocabMapping = Map(
        "i" -> 0,
        "go" -> 1,
        "to" -> 2,
        "the" -> 3,
        "store" -> 4,
        "then" -> 5,
        "movie" -> 6
      )

      val lemmatizer = StanfordLemmatizer()
      val bow = Utils.toBagOfWords(doc, vocabMapping, Some(lemmatizer))

      bow should equal(
        Document(
          Vector(0, 5, 1, 6, 2, 3, 4), Vector(2, 1, 2, 1, 2, 2, 1)
        )
      )

    }

  }

}