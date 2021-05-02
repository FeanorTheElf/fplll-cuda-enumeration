AC_INIT([fplll-cuda-enum], [1.0.0])
AC_CONFIG_MACRO_DIRS([m4])
AM_INIT_AUTOMAKE([foreign subdir-objects])
LT_INIT

AC_LANG_CPLUSPLUS
AC_PROG_CXX

AC_ARG_VAR(NVCC, 
    AS_HELP_STRING([Overrides the location of the cuda compiler])
)

AC_ARG_VAR(CUDA_CFLAGS, 
    AS_HELP_STRING([Contains additional flags to pass to the cuda compiler])
)

AC_ARG_VAR(CUDA_LDLAGS, 
    AS_HELP_STRING([Contains additional flags to pass to the linker when linking cuda libraries])
)

# If the user did not specify NVCC, then set it to the default value nvcc (i.e. we assume
# that the cuda compiler is on PATH)
AS_IF([test "x${NVCC}" = "x" ], [NVCC=nvcc])

AC_OUTPUT([Makefile])