
PENCILCC_REPO           := https://github.com/Meinersbur/pencilcc.git
PENCILCC_COMMIT         := pencilcc

PPCG_EXTRA_OPTIONS=  --no-allow-gnu-extensions

CHANDAN_REPO            := https://github.com/chandangreddy/pencilcc.git
CHANDAN_COMMIT          := pencilcc

SLAMBENCH_REPO          := https://github.com/pamela-project/slambench.git
SLAMBENCH_COMMIT        := master



DATASET                 := /home/toky/work/pamela/peach/living_room_traj2_loop.raw

ROOT_DIR=$(shell pwd)

PENCILCC_INSTALL_DIR=${ROOT_DIR}/pencilcc-install/
CHANDAN_INSTALL_DIR=${ROOT_DIR}/chandan-install/

PPCG_BASE_OPTIONS=
PPCG_BASE_OPTIONS+= --isl-schedule-fuse=min
PPCG_BASE_OPTIONS+= --no-private-memory
PPCG_BASE_OPTIONS+= --no-shared-memory
PPCG_BASE_OPTIONS+= --target=prl
PPCG_BASE_OPTIONS+= --opencl-include-file=cl_kernel_vector.cl
PPCG_BASE_OPTIONS+= --opencl-compiler-options='-I${ROOT_DIR}/src'
PPCG_BASE_OPTIONS+= --no-opencl-print-kernel-types
PPCG_BASE_OPTIONS+= --opencl-embed-kernel-code



# PPCG_BASE_OPTIONS+= --no-allow-gnu-extensions
# PPCG_BASE_OPTIONS+= -v --dump-schedule-constraints --dump-schedule  --dump-final-schedule --dump-sizes --isl-print-stats # verbose PPCG compilation


# PPCG_BASE_OPTIONS+= --struct-pass-by-value # Only work with chandangreddy repo

DEBUG_ENV := # PRL_PROFILING=1 PRL_GPU_PROFILING_DETAILED=1


define PPCG_SIZES
kernel[0]->tile[16,16]; kernel[0]->grid[3000,4000]; kernel[0]->block[16,16];
kernel[1]->tile[16,16]; kernel[1]->grid[3000,4000]; kernel[1]->block[16,16];
kernel[2]->tile[16,16]; kernel[2]->grid[1024,1024]; kernel[2]->block[16,16];
kernel[3]->tile[4,4]; kernel[3]->grid[1024,1024]; kernel[3]->block[4,4];
kernel[4]->tile[4,16]; kernel[4]->grid[3000,4000]; kernel[4]->block[4,16];
kernel[5]->tile[4,16]; kernel[5]->grid[3000,4000]; kernel[5]->block[4,16];
kernel[6]->tile[4,16]; kernel[6]->grid[3000,4000]; kernel[6]->block[4,16];
kernel[7]->tile[8,32]; kernel[7]->grid[3000,4000]; kernel[7]->block[8,32];
kernel[8]->tile[8,32]; kernel[8]->grid[3000,4000]; kernel[8]->block[8,32];
kernel[9]->tile[8,32]; kernel[9]->grid[3000,4000]; kernel[9]->block[8,32];
kernel[10]->tile[4,16]; kernel[10]->grid[3000,4000]; kernel[10]->block[4,16];
kernel[11]->tile[4,4]; kernel[11]->grid[1024,1024]; kernel[11]->block[4,4];
kernel[12]->tile[4,16]; kernel[12]->grid[3000,4000]; kernel[12]->block[4,16];
kernel[13]->tile[16,16]; kernel[13]->grid[3000,4000]; kernel[13]->block[16,16];
kernel[14]->tile[4,16]; kernel[14]->grid[3000,4000]; kernel[14]->block[4,16];
kernel[15]->tile[16,16]; kernel[15]->grid[1000,4000]; kernel[15]->block[16,16];
kernel[16]->tile[16,16]; kernel[16]->grid[1000,4000]; kernel[16]->block[16,16]
endef

all   : build run
build : slambench/build/kfusion/kfusion-main-pencilCL
run   : benchmark.2.pencilcc.log

## RUN ##

benchmark.2.chandan.log  :   slambench-chandan/build/kfusion/kfusion-benchmark-pencilCL slambench-chandan/living_room_traj2_loop.raw
	${DEBUG_ENV} LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CHANDAN_INSTALL_DIR}/lib  ${ROOT_DIR}/$< -s 4.8 -p 0.34,0.5,0.24 -z 4-c 2 -r 1 -k 481.2,480,320,240 -i ${ROOT_DIR}/slambench-chandan/living_room_traj2_loop.raw -o $@

benchmark.2.pencilcc.log  :   slambench/build/kfusion/kfusion-benchmark-pencilCL slambench/living_room_traj2_loop.raw
	${DEBUG_ENV} LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PENCILCC_INSTALL_DIR}/lib  ${ROOT_DIR}/$< -s 4.8 -p 0.34,0.5,0.24 -z 4-c 2 -r 1 -k 481.2,480,320,240 -i ${ROOT_DIR}/slambench/living_room_traj2_loop.raw -o $@

main.2.chandan.log  :   slambench-chandan/build/kfusion/kfusion-main-pencilCL slambench-chandan/living_room_traj2_loop.raw
	${DEBUG_ENV} LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CHANDAN_INSTALL_DIR}/lib  ${ROOT_DIR}/$< -s 4.8 -p 0.34,0.5,0.24 -z 4-c 2 -r 1 -k 481.2,480,320,240 -i ${ROOT_DIR}/slambench-chandan/living_room_traj2_loop.raw -o $@

main.2.pencilcc.log  :   slambench/build/kfusion/kfusion-main-pencilCL slambench/living_room_traj2_loop.raw
	${DEBUG_ENV} LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PENCILCC_INSTALL_DIR}/lib  ${ROOT_DIR}/$< -s 4.8 -p 0.34,0.5,0.24 -z 4-c 2 -r 1 -k 481.2,480,320,240 -i ${ROOT_DIR}/slambench/living_room_traj2_loop.raw -o $@

## SLAMBENCH REPOS ##


slambench-chandan :
	git clone ${SLAMBENCH_REPO} $@
	cd $@ && git checkout ${SLAMBENCH_COMMIT} 
	echo "if (OPENCL_FOUND AND DEFINED ENV{PENCIL_UTIL_HOME} AND DEFINED ENV{PRL_PATH})" >> $@/kfusion/CMakeLists.txt
	echo "set(prl_path \$$ENV{PRL_PATH})" >> $@/kfusion/CMakeLists.txt
	echo "set(util_path \$$ENV{PENCIL_UTIL_HOME})" >> $@/kfusion/CMakeLists.txt
	echo "include_directories(\$${OPENCL_INCLUDE_DIRS} \$${util_path}/include)" >> $@/kfusion/CMakeLists.txt
	echo "add_library(ppcgKernelsHost src/pencil/pencil_kernels_host.c)" >> $@/kfusion/CMakeLists.txt
	echo "SET_TARGET_PROPERTIES(ppcgKernelsHost PROPERTIES COMPILE_FLAGS \"-std=c99 -I\$${util_path}/include\")" >> $@/kfusion/CMakeLists.txt
	echo "add_library(\$${appname}-pencilCL  src/pencil/kernels.cpp)" >> $@/kfusion/CMakeLists.txt
	echo "SET_TARGET_PROPERTIES(\$${appname}-pencilCL PROPERTIES COMPILE_FLAGS \"-I\$${util_path}/include\")" >> $@/kfusion/CMakeLists.txt
	echo "target_link_libraries(\$${appname}-pencilCL   \$${common_libraries} \$${OPENCL_LIBRARIES} ppcgKernelsHost \"-L\$${prl_path}/lib -lprl_opencl\")" >> $@/kfusion/CMakeLists.txt
	echo "add_version(\$${appname} pencilCL \"-D __PENCIL__  -I\$${util_path}/include\" \"-L\$${prl_path}/lib -lprl_opencl\")" >> $@/kfusion/CMakeLists.txt
	echo "endif()" >> $@/kfusion/CMakeLists.txt

slambench-chandan/kfusion/src/pencil/pencil_kernels_host.c :  slambench-chandan chandan-install/bin/ppcg src/pencil_kernels.c
	mkdir -p  slambench-chandan/kfusion/src/pencil/ 
	cd slambench-chandan/kfusion/src/pencil/ && ${CHANDAN_INSTALL_DIR}/bin/ppcg ${PPCG_BASE_OPTIONS} --sizes="{$${PPCG_SIZES}}" ${ROOT_DIR}/src/pencil_kernels.c > ppcg.log 2>&1
	cat  slambench-chandan/kfusion/src/pencil/ppcg.log

slambench-chandan/kfusion/src/pencil/kernels.cpp  : slambench-chandan src/kernels.cpp
	mkdir -p  slambench-chandan/kfusion/src/pencil/ 
	cp src/kernels.cpp slambench-chandan/kfusion/src/pencil/

slambench-chandan/living_room_traj2_loop.raw : slambench-chandan slambench-chandan/build/kfusion/kfusion-benchmark-pencilCL
	if [ -f ${DATASET} ] ; then cp ${DATASET} slambench-chandan ; else make -C slambench-chandan living_room_traj2_loop.raw ; fi


slambench-chandan/build/kfusion/% : slambench-chandan/kfusion/src/pencil/kernels.cpp slambench-chandan/kfusion/src/pencil/pencil_kernels_host.c
	PRL_PATH=${CHANDAN_INSTALL_DIR} PENCIL_UTIL_HOME=${CHANDAN_INSTALL_DIR} CC=${ARCH}gcc CXX=${ARCH}g++ OCLROOT=${OPENCL_SDK} make -C slambench-chandan SPECIFIC_TARGET=$*


slambench :
	git clone ${SLAMBENCH_REPO} $@
	cd $@ && git checkout ${SLAMBENCH_COMMIT} 
	echo "if (OPENCL_FOUND AND DEFINED ENV{PENCIL_UTIL_HOME} AND DEFINED ENV{PRL_PATH})" >> $@/kfusion/CMakeLists.txt
	echo "set(prl_path \$$ENV{PRL_PATH})" >> $@/kfusion/CMakeLists.txt
	echo "set(util_path \$$ENV{PENCIL_UTIL_HOME})" >> $@/kfusion/CMakeLists.txt
	echo "include_directories(\$${OPENCL_INCLUDE_DIRS} \$${util_path}/include)" >> $@/kfusion/CMakeLists.txt
	echo "file(GLOB pencil_output_files" >> $@/kfusion/CMakeLists.txt
	echo "    \"src/pencil/*.c\"" >> $@/kfusion/CMakeLists.txt
	echo ")" >> $@/kfusion/CMakeLists.txt
	echo "add_library(ppcgKernelsHost \$${pencil_output_files})" >> $@/kfusion/CMakeLists.txt
	echo "SET_TARGET_PROPERTIES(ppcgKernelsHost PROPERTIES COMPILE_FLAGS \"-std=c99 -I\$${util_path}/include\")" >> $@/kfusion/CMakeLists.txt
	echo "add_library(\$${appname}-pencilCL  src/pencil/kernels.cpp)" >> $@/kfusion/CMakeLists.txt
	echo "SET_TARGET_PROPERTIES(\$${appname}-pencilCL PROPERTIES COMPILE_FLAGS \"-I\$${util_path}/include\")" >> $@/kfusion/CMakeLists.txt
	echo "target_link_libraries(\$${appname}-pencilCL   \$${common_libraries} \$${OPENCL_LIBRARIES} ppcgKernelsHost \"-L\$${prl_path}/lib -lprl_opencl\")" >> $@/kfusion/CMakeLists.txt
	echo "add_version(\$${appname} pencilCL \"-D __PENCIL__  -I\$${util_path}/include\" \"-L\$${prl_path}/lib -lprl_opencl\")" >> $@/kfusion/CMakeLists.txt
	echo "endif()" >> $@/kfusion/CMakeLists.txt





slambench/kfusion/src/pencil/pencil_kernels_host.c :  slambench pencilcc-install/bin/ppcg src/pencil_kernels.c
	mkdir -p  slambench/kfusion/src/pencil/ 
	cd slambench/kfusion/src/pencil/ && ${PENCILCC_INSTALL_DIR}/bin/ppcg ${PPCG_BASE_OPTIONS} ${PPCG_EXTRA_OPTIONS} --sizes="{$${PPCG_SIZES}}" ${ROOT_DIR}/src/pencil_kernels.c

slambench/kfusion/src/pencil/kernels.cpp  : slambench src/kernels.cpp
	mkdir -p  slambench/kfusion/src/pencil/ 
	cp src/kernels.cpp slambench/kfusion/src/pencil/

slambench/build/kfusion/% : slambench/kfusion/src/pencil/kernels.cpp slambench/kfusion/src/pencil/pencil_kernels_host.c
	PRL_PATH=${PENCILCC_INSTALL_DIR} PENCIL_UTIL_HOME=${PENCILCC_INSTALL_DIR} CC=${ARCH}gcc CXX=${ARCH}g++ OCLROOT=${OPENCL_SDK} make -C slambench SPECIFIC_TARGET=$*


slambench/living_room_traj2_loop.raw : slambench slambench/build/kfusion/kfusion-benchmark-pencilCL
	if [ -f ${DATASET} ] ; then cp ${DATASET} slambench ; else make -C slambench living_room_traj2_loop.raw ; fi




### BUILD PENCIL-CC  ###

chandan/configure :
	git clone ${CHANDAN_REPO} chandan
	cd chandan && git checkout ${CHANDAN_COMMIT} 
	cd chandan && git submodule init && git submodule update
	cd chandan/ppcg && git submodule init && git submodule update
	cd chandan/prl && git checkout pencilcc 
	cd chandan && ./autogen.sh

chandan-build/ppcg/.libs/ppcg : chandan/configure
	mkdir chandan-build -p
	cd chandan-build && ${ROOT_DIR}/chandan/configure --prefix=${ROOT_DIR}/chandan-install
	cd chandan-build && make

chandan-install/bin/ppcg : chandan-build/ppcg/.libs/ppcg
	mkdir -p chandan-install
	rm chandan-install/* -rf
	cd chandan-build && make install

pencilcc/configure :
	git clone ${PENCILCC_REPO}
	cd pencilcc && git checkout ${PENCILCC_COMMIT} 
	cd pencilcc && git submodule init && git submodule update
	cd pencilcc/ppcg && git submodule init && git submodule update
	cd pencilcc && ./autogen.sh

pencilcc-build/ppcg/.libs/ppcg : pencilcc/configure
	mkdir pencilcc-build -p
	cd pencilcc-build && ${ROOT_DIR}/pencilcc/configure --prefix=${ROOT_DIR}/pencilcc-install
	cd pencilcc-build && make

pencilcc-install/bin/ppcg : pencilcc-build/ppcg/.libs/ppcg
	mkdir -p pencilcc-install
	rm pencilcc-install/* -rf
	cd pencilcc-build && make install

clean :
	rm *.log pencilcc *-install *-build slambench slambench-* chandan -rf

.PHONY : build clean run all
