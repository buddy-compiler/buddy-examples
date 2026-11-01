// The stack platform: its configs on top of Chipyard and FireChip, which the
// build references in the Chipyard submodule without changing it.
// Chipyard's build uses scalafix, whose settings read this from the root build.
ThisBuild / scalafixScalaBinaryVersion := scalaBinaryVersion.value

lazy val chipyard = ProjectRef(file("thirdparty/chipyard"), "chipyard")
lazy val firechip = ProjectRef(file("thirdparty/chipyard"), "firechip")

lazy val platform = (project in file("."))
  .dependsOn(chipyard, firechip)
  .settings(
    name := "platform",
    scalaVersion := "2.13.16",
    libraryDependencies += "org.chipsalliance" %% "chisel" % "6.7.0",
    addCompilerPlugin("org.chipsalliance" % "chisel-plugin" % "6.7.0" cross CrossVersion.full)
  )
