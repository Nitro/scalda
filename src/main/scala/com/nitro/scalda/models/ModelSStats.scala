package com.nitro.scalda.models

@SerialVersionUID(7334410294989955983L)
case class ModelSStats[T](lambda: T, vocabMapping: Map[String, Int], numUpdates: Int) extends Serializable
