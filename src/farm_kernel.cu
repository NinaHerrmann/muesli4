template <class I, class O, class F>
__global__ void msl::detail::farmKernel(I* input, O* output, int size, F f)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < size) {
    output[i] = f(input[i]);
	}
}
