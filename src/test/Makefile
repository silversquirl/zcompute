SRC = $(wildcard *.comp)
OUT = $(patsubst %.comp,%.spv,${SRC})
all: ${OUT}
%.spv: %.comp
	glslc --target-env=vulkan1.1 -o $@ $<
