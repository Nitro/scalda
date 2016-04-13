package com.nitro.scalda.models

import breeze.linalg.DenseMatrix

trait OnlineLDA {

  type BOWMinibatch
  type MinibatchSStats
  type LdaModel
  type Lambda
  type Minibatch

  type Gamma = DenseMatrix[Double]

  def eStep(mb: BOWMinibatch, lambda: Lambda, gamma: Gamma): MinibatchSStats

  def mStep(model: LdaModel, mbSStats: MinibatchSStats): LdaModel

}

