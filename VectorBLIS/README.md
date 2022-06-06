# Vector BLIS project

## zDNN
 - written mostly in C
 - provides libraries, frameworks, and model compilers to utilize the IBM Z Integrated Accelerator for AI
## BLIS
 - framework for instantiating high performance BLAS libraries
 - allows row-major and general stride
 - can directly call lower-level packing, computation kernels
 - unifies packing for 3 different matrix stuctures into one interface
	- general matrices
	- symmetric/hermitian matrices
	- triangular matrices
## BLAS
 - Basic Linear ALgebra Subprograms
	- Level 1: vector-vector
	- Level 2: matrix-vector
	- Level 3: matrix-matrix
 - only column-major
