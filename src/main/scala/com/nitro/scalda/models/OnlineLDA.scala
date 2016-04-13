package com.nitro.scalda.models

import breeze.linalg.DenseMatrix

trait OnlineLda {

  type BowMinibatch
  type MinibatchSStats
  type LdaModel
  type Lambda
  type Minibatch

  type Gamma = DenseMatrix[Double]

  def eStep(mb: BowMinibatch, lambda: Lambda, gamma: Gamma): MinibatchSStats

  def mStep(model: LdaModel, mbSStats: MinibatchSStats): LdaModel

}