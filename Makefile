
#PENCILCC_REPO           := https://github.com/Meinersbur/pencilcc.git
PENCILCC_REPO           := https://github.com/chandangreddy/pencilcc.git
PENCILCC_COMMIT         := pencilcc

SLAMBENCH_REPO          := https://github.com/pamela-project/slambench.git
SLAMBENCH_COMMIT        := master

DATASET                 := /home/toky/work/pamela/peach/living_room_traj2_loop.raw

ROOT_DIR=$(shell pwd)
INSTALL_DIR=${ROOT_DIR}/pencilcc-install/

PRL_PATH=${INSTALL_DIR}
PENCIL_UTIL_HOME=${INSTALL_DIR}

PPCG_BASE_OPTIONS=
PPCG_BASE_OPTIONS+= --isl-schedule-fuse=min
PPCG_BASE_OPTIONS+= --no-private-memory
PPCG_BASE_OPTIONS+= --no-shared-memory
PPCG_BASE_OPTIONS+= --target=prl
PPCG_BASE_OPTIONS+= --opencl-include-file=cl_kernel_vector.cl
PPCG_BASE_OPTIONS+= --opencl-compiler-options='-I${ROOT_DIR}/src'
PPCG_BASE_OPTIONS+= --no-opencl-print-kernel-types
PPCG_BASE_OPTIONS+= --opencl-embed-kernel-code

# PPCG_BASE_OPTIONS+= --struct-pass-by-value # Only work with chandangreddy repo

DEBUG_ENV := # PRL_PROFILING=1 PRL_GPU_PROFILING_DETAILED=1

PPCG= ${INSTALL_DIR}/bin/ppcg
# PPCG+= -v --dump-schedule-constraints --dump-schedule  --dump-final-schedule --dump-sizes --isl-print-stats # verbose PPCG compilation


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
run   : benchmark.2.PencilCL.log

## RUN ##

benchmark.2.PencilCL.log  :   slambench/build/kfusion/kfusion-main-pencilCL slambench/living_room_traj2_loop.raw
	${DEBUG_ENV} LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PRL_PATH}/lib  ${ROOT_DIR}/slambench/build/kfusion/kfusion-main-pencilCL -s 4.8 -p 0.34,0.5,0.24 -z 4-c 2 -r 1 -k 481.2,480,320,240 -i ${ROOT_DIR}/slambench/living_room_traj2_loop.raw -o ${ROOT_DIR}/benchmark.2.PencilCL.log

## SLAMBENCH REPOS ##

slambench :
	git clone ${SLAMBENCH_REPO}
	cd slambench && git checkout ${SLAMBENCH_COMMIT} 
	echo "if (OPENCL_FOUND AND DEFINED ENV{PENCIL_UTIL_HOME} AND DEFINED ENV{PRL_PATH})" >> slambench/kfusion/CMakeLists.txt
	echo "set(prl_path \$$ENV{PRL_PATH})" >> slambench/kfusion/CMakeLists.txt
	echo "set(util_path \$$ENV{PENCIL_UTIL_HOME})" >> slambench/kfusion/CMakeLists.txt
	echo "include_directories(\$${OPENCL_INCLUDE_DIRS} \$${util_path}/include)" >> slambench/kfusion/CMakeLists.txt
	echo "add_library(ppcgKernelsHost src/pencil/pencil_kernels_host.c src/pencil/pencil_kernels_kernel.c)" >> slambench/kfusion/CMakeLists.txt
	echo "SET_TARGET_PROPERTIES(ppcgKernelsHost PROPERTIES COMPILE_FLAGS \"-std=c99 -I\$${util_path}/include\")" >> slambench/kfusion/CMakeLists.txt
	echo "add_library(\$${appname}-pencilCL  src/pencil/kernels.cpp)" >> slambench/kfusion/CMakeLists.txt
	echo "SET_TARGET_PROPERTIES(\$${appname}-pencilCL PROPERTIES COMPILE_FLAGS \"-I\$${util_path}/include\")" >> slambench/kfusion/CMakeLists.txt
	echo "target_link_libraries(\$${appname}-pencilCL   \$${common_libraries} \$${OPENCL_LIBRARIES} ppcgKernelsHost \"-L\$${prl_path}/lib -lprl_opencl\")" >> slambench/kfusion/CMakeLists.txt
	echo "add_version(\$${appname} pencilCL \"-D __PENCIL__  -I\$${util_path}/include\" \"-L\$${prl_path}/lib -lprl_opencl\")" >> slambench/kfusion/CMakeLists.txt
	echo "endif()" >> slambench/kfusion/CMakeLists.txt


slambench/kfusion/src/pencil/pencil_kernels_host.c :  slambench pencilcc-install/bin/ppcg
	mkdir -p  slambench/kfusion/src/pencil/ 
	cd slambench/kfusion/src/pencil/ && ${PPCG} ${PPCG_BASE_OPTIONS} --sizes="{$${PPCG_SIZES}}" ${ROOT_DIR}/src/pencil_kernels.c

slambench/kfusion/src/pencil/kernels.cpp  : slambench 
	cp src/kernels.cpp slambench/kfusion/src/pencil/

slambench/build/kfusion/kfusion-main-pencilCL : slambench slambench/kfusion/src/pencil/pencil_kernels_host.c slambench/kfusion/src/pencil/kernels.cpp  
	PRL_PATH=${PRL_PATH} PENCIL_UTIL_HOME=${PENCIL_UTIL_HOME} CC=${ARCH}gcc CXX=${ARCH}g++ OCLROOT=${OPENCL_SDK} make -C slambench

slambench/living_room_traj2_loop.raw : slambench
	if [ -f ${DATASET} ] ; then cp ${DATASET} slambench ; else make -C slambench living_room_traj2_loop.raw ; fi


### BUILD PENCIL-CC ###

pencilcc/configure :
	git clone ${PENCILCC_REPO}
	cd pencilcc && git checkout ${PENCILCC_COMMIT} 
	cd pencilcc && git submodule init && git submodule update
	cd pencilcc/ppcg && git submodule init && git submodule update
	cd pencilcc/prl && git checkout pencilcc 
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
	rm benchmark.2.PencilCL.log pencilcc *-install *-build slambench -rf

.PHONY : build clean run all
