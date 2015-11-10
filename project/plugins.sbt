logLevel := Level.Warn

resolvers += ("Nitro Nexus Releases" at "https://nexus.nitroplatform.com/nexus/content/repositories/releases/")

addSbtPlugin("com.gonitro.platform" % "sbt-nitro" % "0.1.0")

addSbtPlugin("org.scoverage" %% "sbt-scoverage" % "0.99.7.1")

addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.14.0")

addSbtPlugin("com.typesafe.sbt" % "sbt-scalariform" % "1.3.0")

addSbtPlugin("org.xerial.sbt" % "sbt-pack" % "0.7.5")