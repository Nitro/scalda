package com.nitro.scalda.models

case class MbSStats[T, U](topicUpdates: T, topicProportions: U, mbSize: Int = 0)
