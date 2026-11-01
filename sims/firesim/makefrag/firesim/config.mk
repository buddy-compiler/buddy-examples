# Thin stack platform configuration for the upstream FireSim build

# Only used in this projects makefrags
makefile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
makefile_dir := $(patsubst %/,%,$(dir $(makefile_path)))
chipyard_dir := $(abspath $(makefile_dir)/../../../../platforms/thirdparty/chipyard)

# These point at the main class of the target's Chisel generator
DESIGN_PACKAGE ?= firechip.chip
DESIGN ?= FireSim

# Override to use our custom config package
TARGET_CONFIG_PACKAGE ?= platform

# These guide chisel elaboration of simulation components by MIDAS,
# including models and widgets.
PLATFORM_CONFIG_PACKAGE ?= firesim.configs

# Override project for the target.
TARGET_SBT_PROJECT := platform

# The stack platform build, which references Chipyard
TARGET_SBT_DIR := $(abspath $(makefile_dir)/../../../../platforms)
TARGET_SOURCE_DIRS := $(abspath $(makefile_dir)/../../../../platforms/src/main/scala)

# SBT launcher
SBT ?= $(JAVA_HOME)/bin/java -Xmx8G -jar $(chipyard_dir)/scripts/sbt-launch.jar $(SBT_OPTS)
