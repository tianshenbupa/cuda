```cpp
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#define N 10000000  // 向量大小：1000 万
#define BLOCK_SIZE_1D 1024  // 1D 线程块大小
#define BLOCK_SIZE_3D_X 16  // 3D 线程块在 x 方向的维度
#define BLOCK_SIZE_3D_Y 8   // 3D 线程块在 y 方向的维度
#define BLOCK_SIZE_3D_Z 8   // 3D 线程块在 z 方向的维度（16*8*8=1024 线程）

// CPU 实现的向量加法
void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// GPU 实现的 1D 向量加法内核
__global__ void vector_add_gpu_1d(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 计算全局线程索引
    if (i < n) {
        c[i] = a[i] + b[i];  // 每个线程计算一个元素的加法
    }
}

// GPU 实现的 3D 向量加法内核
__global__ void vector_add_gpu_3d(float *a, float *b, float *c, int nx, int ny, int nz) {
    // 三维线程索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        // 将 3D 索引映射为 1D 索引
        int idx = i + j * nx + k * nx * ny;
        if (idx < nx * ny * nz) {
            c[idx] = a[idx] + b[idx];
        }
    }
}

// 向量初始化为随机浮点数
void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

// 获取当前时间（秒）
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    // 主机内存指针
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu_1d, *h_c_gpu_3d;

    // 设备内存指针
    float *d_a, *d_b, *d_c_1d, *d_c_3d;

    // 分配大小
    size_t size = N * sizeof(float);

    // 分配主机内存
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu_1d = (float*)malloc(size);
    h_c_gpu_3d = (float*)malloc(size);

    // 初始化输入向量
    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    // 分配设备内存
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c_1d, size);
    cudaMalloc(&d_c_3d, size);

    // 将主机数据复制到设备
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 配置 GPU 1D 网格大小
    int num_blocks_1d = (N + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;

    // 配置 GPU 3D 网格大小
    int nx = 100, ny = 100, nz = 1000;  // nx * ny * nz = 10^7
    dim3 block_size_3d(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);
    dim3 num_blocks_3d(
        (nx + block_size_3d.x - 1) / block_size_3d.x,
        (ny + block_size_3d.y - 1) / block_size_3d.y,
        (nz + block_size_3d.z - 1) / block_size_3d.z
    );

    // 预热运行，避免首次执行带来的延迟
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);  // CPU
        vector_add_gpu_1d<<<num_blocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_1d, N);  // GPU 1D
        vector_add_gpu_3d<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c_3d, nx, ny, nz);  // GPU 3D
        cudaDeviceSynchronize();
    }

    // CPU 基准测试
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 5; i++) {
        double start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 5.0;

    // GPU 1D 基准测试
    printf("Benchmarking GPU 1D implementation...\n");
    double gpu_1d_total_time = 0.0;
    for (int i = 0; i < 100; i++) {
        cudaMemset(d_c_1d, 0, size);  // 清空结果内存
        double start_time = get_time();
        vector_add_gpu_1d<<<num_blocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_1d, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_1d_total_time += end_time - start_time;
    }
    double gpu_1d_avg_time = gpu_1d_total_time / 100.0;

    // 校验 GPU 1D 结果
    cudaMemcpy(h_c_gpu_1d, d_c_1d, size, cudaMemcpyDeviceToHost);
    bool correct_1d = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu_1d[i]) > 1e-4) {
            correct_1d = false;
            std::cout << i << " cpu: " << h_c_cpu[i] << " != " << h_c_gpu_1d[i] << std::endl;
            break;
        }
    }
    printf("1D Results are %s\n", correct_1d ? "correct" : "incorrect");

    // GPU 3D 基准测试
    printf("Benchmarking GPU 3D implementation...\n");
    double gpu_3d_total_time = 0.0;
    for (int i = 0; i < 100; i++) {
        cudaMemset(d_c_3d, 0, size);  // 清空结果内存
        double start_time = get_time();
        vector_add_gpu_3d<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c_3d, nx, ny, nz);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_3d_total_time += end_time - start_time;
    }
    double gpu_3d_avg_time = gpu_3d_total_time / 100.0;

    // 校验 GPU 3D 结果
    cudaMemcpy(h_c_gpu_3d, d_c_3d, size, cudaMemcpyDeviceToHost);
    bool correct_3d = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu_3d[i]) > 1e-4) {
            correct_3d = false;
            std::cout << i << " cpu: " << h_c_cpu[i] << " != " << h_c_gpu_3d[i] << std::endl;
            break;
        }
    }
    printf("3D Results are %s\n", correct_3d ? "correct" : "incorrect");

    // 输出性能测试结果
    printf("CPU average time: %f milliseconds\n", cpu_avg_time * 1000);
    printf("GPU 1D average time: %f milliseconds\n", gpu_1d_avg_time * 1000);
    printf("GPU 3D average time: %f milliseconds\n", gpu_3d_avg_time * 1000);
    printf("Speedup (CPU vs GPU 1D): %fx\n", cpu_avg_time / gpu_1d_avg_time);
    printf("Speedup (CPU vs GPU 3D): %fx\n", cpu_avg_time / gpu_3d_avg_time);
    printf("Speedup (GPU 1D vs GPU 3D): %fx\n", gpu_1d_avg_time / gpu_3d_avg_time);

    // 释放内存
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu_1d);
    free(h_c_gpu_3d);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_1d);
    cudaFree(d_c_3d);

    return 0;
}


---
---
---



以下是你提供的 **CUDA 向量加法程序分析** 按照你所需的结构整理、提炼和补充，供你参考或作为教学/文档使用。

---

## 🧩 **CUDA 向量加法程序概述**

该程序完成了两个长度为 **1 千万（10⁷）** 的 `float` 向量逐元素加法操作，分别使用 **CPU** 和两种 **GPU 内核实现（1D 和 3D）**，并进行：

* **性能基准测试**（Benchmark）
* **结果正确性验证**
* **加速比分析**

---

## 🔩 **宏与参数设定**

```cpp
#define N 10000000         // 向量长度
#define BLOCK_SIZE_1D 1024 // 1D 线程块大小
#define BLOCK_SIZE_3D_X 16
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 8  // 每个 3D 线程块共 16*8*8=1024 线程
```

* 用 3D block 模拟 volume grid：`100 * 100 * 1000 = 10⁷`

---

## 🧠 **主要函数**

### ✅ `vector_add_cpu`

> 使用经典的逐元素循环：

```cpp
for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
}
```

用于基准结果和正确性验证。

---

### ✅ `__global__ void vector_add_gpu_1d`

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
    c[i] = a[i] + b[i];
}
```

* **线性索引计算**
* 每个线程处理一个 `a[i] + b[i]`
* 加边界保护防止越界访问

---

### ✅ `__global__ void vector_add_gpu_3d`

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;
int k = blockIdx.z * blockDim.z + threadIdx.z;
int idx = i + j * nx + k * nx * ny;
```

* 显式计算 `3D → 1D` 映射下标 `idx`
* 最后访问 `a[idx] + b[idx]`，写入 `c[idx]`
* 添加 `idx < N` 检查以防冗余线程执行无效访问

---

## 🧪 **时间测量与执行逻辑**

### ✅ `double get_time()`

使用 `clock_gettime(CLOCK_MONOTONIC)` 实现高精度计时。

> 更推荐使用 CUDA 原生计时：`cudaEventRecord()` （下面会建议改进）

---

### ✅ Warm-up

```cpp
for (int i = 0; i < 3; i++) {
    ...
    cudaDeviceSynchronize();
}
```

* 避免首次运行带来的延迟（如 kernel JIT 编译、context 初始化）

---

### ✅ Benchmark 测试逻辑

* CPU：执行 5 次取平均
* GPU：执行 100 次取平均（更稳定）

```cpp
cudaMemset(...)     // 清零结果区
kernel<<<...>>>()   // 执行 kernel
cudaDeviceSynchronize();
```

---

## 🧪 **正确性验证**

```cpp
fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-4
```

* 控制在 `1e-4` 的单精度误差范围
* 遇错即停止输出差值（调试用）

---

## 📈 **输出性能指标**

```text
CPU average time: ...
GPU 1D average time: ...
GPU 3D average time: ...
Speedup (CPU vs GPU 1D): ...
```

**说明：**

* 对比 CPU 与 GPU 加速比
* 同时比较 1D 与 3D 配置的 GPU 效率差异

---

## 🚀 **进阶优化建议**

| 方向            | 优化建议                                                            |
| ------------- | --------------------------------------------------------------- |
| ⏱️ 计时精度       | 使用 `cudaEvent_t` 替代 `clock_gettime`，专测 kernel 执行时间，避免主机干扰       |
| 💾 数据传输       | 使用 pinned memory（页锁定内存）提高 H2D/D2H 带宽                            |
| 🔁 异步执行       | 利用 CUDA stream 实现 memcpy 与 kernel 的 overlap                     |
| 📐 block/grid | 动态自适应 blockSize，或实验 2D grid 性能表现                                |
| 🧮 内核逻辑       | 添加 shared memory 优化/避免 bank conflict（适合更复杂运算）                   |
| 📊 性能分析       | 使用 `nvprof` 或 Nsight Compute 查看 memory throughput、occupancy 等指标 |
| 🧬 模板化扩展      | 模板支持 `float`、`double` 类型兼容（for scientific computing）            |

---

## 🎯 **改进版时间测量（推荐）**

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<grid, block>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms = 0;
cudaEventElapsedTime(&ms, start, stop);
```

* **单位为毫秒（ms）**
* 排除 Host 控制时间，只测 kernel 部分
* 对短内核尤其重要

---

## ✅ 总结知识点

| 概念     | 涉及位置                                    |
| ------ | --------------------------------------- |
| 一维线程索引 | `blockIdx.x * blockDim.x + threadIdx.x` |
| 三维线程索引 | `i + j * nx + k * nx * ny`              |
| 网格维度计算 | `(N + BLOCK_SIZE - 1) / BLOCK_SIZE`     |
| 主机设备内存 | `malloc / cudaMalloc / cudaMemcpy`      |
| 结果验证   | `fabs(cpu - gpu) > ε`                   |
| 性能统计   | 取平均 + speedup 输出                        |

---

是否需要我帮你：

* ✍ 改写一个使用 `cudaEvent` 的版本？
* 🧪 扩展为 vector multiply / dot product？
* 📊 生成 Excel/CSV 格式的基准测试结果方便绘图？
* 📌 编写一个用于 Nsight Profiler 分析的 wrapper 脚本？

可以继续深挖内核调优、CUDA Occupancy 分析或实际应用场景（如推荐系统或金融计算）拓展。你想继续哪个方向？








