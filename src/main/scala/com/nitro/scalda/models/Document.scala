package com.nitro.scalda.models

case class Document(
  wordIds: IndexedSeq[Int],
  wordCts: IndexedSeq[Int]
)
