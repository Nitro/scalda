package com.nitro.scalda

import com.nitro.scalda.tokenizer.StanfordLemmatizer
import org.scalatest.{ Matchers, WordSpec }

class StanfordLemmatizerSpec extends WordSpec with Matchers {

  "The stanford lemmatizer" should {

    "lemmatize sentences correctly" in {

      val testText = "Today I went to the store. I also ran to the cinema."

      val lemmas = StanfordLemmatizer().tokenize(testText)

      lemmas should equal(Seq("today", "I", "go", "to", "the", "store", ".", "I", "also", "run", "to", "the", "cinema", "."))
    }

  }

}