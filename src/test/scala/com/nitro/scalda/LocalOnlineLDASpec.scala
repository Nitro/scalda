package com.nitro.scalda

import breeze.linalg.DenseMatrix
import com.nitro.scalda.models.onlineLDA.local.LocalOnlineLda
import com.nitro.scalda.models.{ ModelSStats, Document, OnlineLdaParams }
import org.scalatest.{ Matchers, WordSpec }

class LocalOnlineLdaSpec extends WordSpec with Matchers {

  "Local FS based Online LDA" should {

    val doc = Document(
      IndexedSeq(0, 1, 2, 3), //ids
      IndexedSeq(5, 2, 1, 8) //cts
    )

    val testLambda = DenseMatrix(
      (0.22, 0.7, 0.2, 0.3),
      (0.14, 0.29, 0.5, 0.2),
      (0.32, 0.43, 0.2, 0.45)
    )

    val testGamma = DenseMatrix((0.11, 0.22, 0.11))

    val testVocab = IndexedSeq("hello", "world", "foo", "bar")

    //test parameters
    val params = OnlineLdaParams(
      vocabulary = testVocab,
      alpha = 1.0 / 3,
      eta = 1.0 / 3,
      decay = 1024,
      learningRate = 0.7,
      maxIter = 100,
      convergenceThreshold = 0.001,
      numTopics = 3,
      totalDocs = 5220
    )

    val testLDA = LocalOnlineLda(params)

    "compute the correct E-Step" in {

      val mbSStats = testLDA.eStep(
        mb = Seq(doc),
        lambda = testLambda,
        gamma = testGamma
      )

      mbSStats.topicUpdates should equal(
        DenseMatrix(
          (0.004337997523699932, 0.024578067243257937, 0.001775775187582318, 0.009064515117168562),
          (0.0032959055127366158, 0.025017135815613923, 0.5590950723373326, 0.01719131457322387),
          (4.992366096963564, 1.950404796941128, 0.43912915247508516, 7.973744170309609)
        )
      )

      mbSStats.topicProportions should equal(
        DenseMatrix((0.3731066102557286, 0.9383461525508197, 15.688547237193452))
      )

    }

    "compute the correct m-step" in {

      val testMbSStats = testLDA.eStep(
        mb = Seq(doc),
        lambda = testLambda,
        gamma = testGamma
      )

      val testMapping = testVocab.zipWithIndex.toMap

      val oldModelSStats = ModelSStats(
        lambda = testLambda,
        vocabMapping = testMapping,
        numUpdates = 4
      )

      val newModelSStats = testLDA.mStep(
        model = oldModelSStats,
        mbSStats = testMbSStats
      )

      newModelSStats.lambda should equal(DenseMatrix(
        (0.3972803623230393, 1.6965800010534808, 0.27327901193597176, 0.6688533802521014),
        (0.2754473254960829, 1.3070105783689454, 23.229391163361104, 0.8996708276963741),
        (203.36030412227188, 79.75335256242374, 18.068216670222686, 324.7427271499273)
      ))

      newModelSStats.numUpdates should equal(5)

    }

    /*

    "compute the correct perplexity" in {

      val testPerplexity = testLDA.perplexity(
        mb = Seq(doc),
        mbGamma = testGamma,
        lambda = testLambda
      )

      testPerplexity should equal(-238735.18879931796)
    }


    */
  }

}