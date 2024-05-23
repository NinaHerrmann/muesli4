template <typename T>
class Foo {
public:
    MSL_USERFUNC Foo() {}
    MSL_USERFUNC ~Foo() {}
    MSL_USERFUNC int get(int a, int b) {return a;}
    __device__ void setdata( int r, int c, int firstRow, int cols, int width, int stencil_size) {
        T *smem = SharedMemory<T>();
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = r-firstRow;

        /*smem[((ty + stencil_size) * width) + tx + stencil_size] = current_data[(row) * cols + c];

        // left side
        if (c == 0 || (c%(width-(2*stencil_size)))==0) {
            smem[(ty + stencil_size) * width + tx + stencil_size -1] = current_data[(row) * cols +c - 1];
        }
        // right side
        if ((c+1+(2*stencil_size))%width == 0) {
            smem[(ty + stencil_size) * width + tx + stencil_size + 1] = current_data[(row) * cols +c + 1];
        }
        // first row
        if (r == 0 || (r%(width-(2*stencil_size)))==0) {
            smem[(ty+stencil_size-1)*width + tx+stencil_size] = current_data[(row-1)*cols + c];
        }
        // last row
        if (r == (width-1-2*stencil_size)) {
            smem[(ty + stencil_size+1) * width + tx + stencil_size] = current_data[(row + 1) * cols + c];
        }*/

        __syncthreads();

        shared_data = smem;
        }

public:
    T* shared_data;
    int test = 1;
};