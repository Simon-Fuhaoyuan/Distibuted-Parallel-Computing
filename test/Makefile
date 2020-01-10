all : test.cu
	nvcc test.cu -o test.out -lcublas

vec : test_mat.cu
	nvcc test_mat.cu -o vec.out -lcublas

clean :
	rm *.out
