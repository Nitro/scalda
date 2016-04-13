package com.nitro.scalda.models

case class OnlineLdaParams(
  vocabulary: IndexedSeq[String],
  alpha: Double,
  eta: Double,
  decay: Double,
  learningRate: Double,
  maxIter: Int,
  convergenceThreshold: Double,
  numTopics: Int,
  totalDocs: Int,
  lemmatize: Boolean = false,
  perplexity: Boolean = false
)