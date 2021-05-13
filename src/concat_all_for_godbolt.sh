rm out.cu
echo "#include \"cooperative_groups.h\"//" >> out.cu
echo "#include \"cuda_runtime.h\"//" >> out.cu
echo "#include \"device_launch_parameters.h\"//" >> out.cu
cat constants.cuh >> out.cu
cat atomic.h >> out.cu
cat cuda_check.h >> out.cu
cat memory.h >> out.cu
cat cuda_util.cuh >> out.cu
cat prefix.cuh >> out.cu
cat types.cuh >> out.cu
cat streaming.cuh >> out.cu
cat recenum.cuh >> out.cu
cat enum.cuh >> out.cu
cat api.h >> out.cu
cat cuda_wrapper.h >> out.cu
cat cuda_wrapper_out.cu >> out.cu
sed -i 's/#include\s"[^"]*"[^\/]//g' out.cu