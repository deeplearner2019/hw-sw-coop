ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:$(PYTHONPATH)
endif

NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck

MODEL_DIR =
LOG_FILE = 

MODEL_FILENAME = ${MODEL_DIR}frozen_graph.pb
GRAPH_FILENAME = ${MODEL_DIR}frozen_graph.graph
#CONV_SCRIPT = ./save_session.py

INPUT_NODE_FLAG = input
OUTPUT_NODE_FLAG = CifarNet/Predictions/Reshape_1
#OUTPUT_NODE_FLAG = -on=final_layer/predictions

COMPILE_FULL_COMMAND = ${NCCOMPILE} ${MODEL_FILENAME} -s 12 -in ${INPUT_NODE_FLAG} -on ${OUTPUT_NODE_FLAG} -o ${GRAPH_FILENAME}

IMGSIZE = 32

.PHONY: all
all: compile profile
	@echo "\nlaunching routine\n"

.PHONY: compile
compile:
	@echo "\nmaking compile" ; \
	if [ -e ${GRAPH_FILENAME} ] ; \
	then \
		echo "graph file exists, skipping compilation." ; \
		echo "    if you want to re-compile, remove ${GRAPH_FILENAME}, and re-run" ; \
	else \
		#cd ${MODEL_DIR_NCS} ; \
		echo "Command line: ${COMPILE_FULL_COMMAND}" ; \
		${COMPILE_FULL_COMMAND} ; \
		cp ${GRAPH_FILENAME} ../.. ; \
		cd ../.. ; \
	fi

.PHONY: profile
profile:
	@echo $(YELLOW)"\nProfiling the model..."$(NOCOLOR)
	mvNCProfile -s 12 ${MODEL_FILENAME} -in ${INPUT_NODE_FLAG} -on ${OUTPUT_NODE_FLAG} >${LOG_FILE} 
	@echo $(YELLOW)"Movidius graph generated! You can run inferences using ncappzoo/apps/image-classifier project.\n"$(NOCOLOR)

.PHONY: check
check:
	@echo $(YELLOW)"\nComparing results with standard Tensorflow..."$(NOCOLOR)
	mvNCCheck -s 12 ${MODEL_FILENAME} -in=${INPUT_NODE_FLAG} -on=${OUTPUT_NODE_FLAG}
	@echo $(YELLOW)"Movidius graph generated! You can run inferences using ncappzoo/apps/image-classifier project.\n"$(NOCOLOR)

.PHONY: run
run: compile
	@echo $(YELLOW)"\nRunning inferences using image-classifier project..."$(NOCOLOR)
	(python3 ../../apps/image-classifier/image-classifier.py --graph ./${GRAPH_FILENAME} --dim $(IMGSIZE) $(IMGSIZE) --mean 127.5 --scale 0.00789 --colormode "RGB" --labels ./labels_cifar10.txt --image ../cifar10_images/number0.jpg)

.PHONY: help
help:
	@echo $(YELLOW)"\nPossible make targets: "$(NOCOLOR);
	@echo "  make help - Shows this message.";
	@echo "  make all - makes the following: compile, profile";
	@echo "  make clean - Removes all files created in this project.;"
	@echo "  make compile - Convert the frozen model into Movidius graph file.";
	@echo "  make check - Compare inference results with that of TensorFlow running on CPU/GPU.";
	@echo "  make profile - Run the model on NCS and extract complexity, bandwidth and execution time for each layer.";
