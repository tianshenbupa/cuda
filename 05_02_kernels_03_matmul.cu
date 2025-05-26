下面是你提供的 CUDA 矩阵乘法代码，**已添加详细中文注释**，包括 CPU/GPU 实现、初始化、计时等部分，适合学习和调试使用：

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// 定义矩阵维度
#define M 256  // A 和 C 的行数
#define K 512  // A 的列数，B 的行数
#define N 256  // B 和 C 的列数
#define BLOCK_SIZE 32  // CUDA 每个线程块的尺寸

// ==================== CPU 实现矩阵乘法 ====================
void matmul_cpu(float *A, float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {         // 遍历 C 的每一行
        for (int j = 0; j < n; j++) {     // 遍历 C 的每一列
            float sum = 0.0f;
            for (int l = 0; l < k; l++) { // 内积运算
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// ==================== GPU 核函数实现矩阵乘法 ====================
__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 当前线程计算的行号
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 当前线程计算的列号

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++) {
            sum += A[row * k + l] * B[l * n + col]; // 执行内积
        }
        C[row * n + col] = sum; // 写入结果
    }
}

// ==================== 随机初始化矩阵元素 ====================
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX; // 随机浮点数 [0, 1)
    }
}

// ==================== 获取当前系统时间（秒） ====================
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts); // 高精度计时
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ==================== 主函数入口 ====================
int main() {
    // 主机端指针
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;

    // 设备端指针
    float *d_A, *d_B, *d_C;

    // 计算所需内存大小
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    // 分配主机内存
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C_cpu = (float*)malloc(size_C);
    h_C_gpu = (float*)malloc(size_C);

    // 初始化输入矩阵
    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // 分配设备内存
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // 将数据从主机复制到设备
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // 定义线程块维度和网格维度（覆盖所有输出元素）
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);  // 每个线程块 BLOCK_SIZE x BLOCK_SIZE
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);  // 保证所有元素被覆盖

    // ========== 预热（warm-up）运行，提升后续测量稳定性 ==========
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N); // CPU 运行
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N); // GPU 运行
        cudaDeviceSynchronize(); // 等待 GPU 执行完毕
    }

    // ========== 测量 CPU 执行时间 ==========
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    // ========== 测量 GPU 执行时间 ==========
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize(); // 保证时间测量准确
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    // ========== 输出性能结果 ==========
    printf("CPU average time: %f microseconds\n", cpu_avg_time * 1e6);
    printf("GPU average time: %f microseconds\n", gpu_avg_time * 1e6);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    // ========== 释放内存 ==========
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
```

---

### ✅ 注释重点提示：

* **坐标计算方式**：

  ```cpp
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  ```

* **核函数要加边界判断**，防止非法访问。

* **时间测量使用 `clock_gettime`**，比 `clock()` 更准确，适合微秒级测量。

---

如果你接下来想要：

* 加入 **共享内存优化版本**；
* 使用 **cuBLAS 加速对比**；
* 加入 **验证 CPU 与 GPU 结果是否一致**；

我都可以帮你补充。需要哪一部分，可以直接告诉我。
