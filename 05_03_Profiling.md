当然，以下是你提供内容的整理与**纯介绍性总结**，专注于工具用途、操作命令、适用场景，不包含任何分析或评估建议：

---

## 🧩 CUDA Kernel Profiling 工具与流程介绍

### 1. **CUDA 程序编译与 Profile 基础**

使用 `nvcc` 编译 CUDA 程序：

```bash
nvcc -o 00 00\ nvtx_matmul.cu -lnvToolsExt
nvcc -o 01 01_naive_matmul.cu
nvcc -o 02 02_tiled_matmul.cu
```

使用 `nsys`（Nsight Systems）收集运行时性能数据：

```bash
nsys profile --stats=true ./00
nsys profile --stats=true ./01
nsys profile --stats=true ./02
```

也可以用于 Python 脚本：

```bash
nsys profile --stats=true -o mlp python mlp.py
```

---

### 2. **生成的 Profile 文件类型**

| 文件类型        | 用法                                      |
| ----------- | --------------------------------------- |
| `.nsys-rep` | 用于 Nsight Systems GUI 分析，可查看时间线、事件、流等信息 |
| `.sqlite`   | 用于 `nsys analyze` 进行结构化分析，支持 SQL 查询     |

---

### 3. **常见命令行工具**

| 工具名                 | 功能描述                                     |
| ------------------- | ---------------------------------------- |
| `nvidia-smi`        | 显示 GPU 利用率、温度、显存等，常与 `watch -n 0.1` 搭配使用 |
| `nvitop`            | 类似 `htop`，可实时显示 GPU 使用状态                 |
| `compute-sanitizer` | 检查 CUDA 程序中的内存越界、未初始化内存等问题               |

---

### 4. **Nsight 系列工具定位与使用**

| 工具名                     | 功能描述                                  |
| ----------------------- | ------------------------------------- |
| `nsys` (Nsight Systems) | 系统级 profile 工具，宏观查看 kernel 调度、流、时间占比等 |
| `ncu` (Nsight Compute)  | 内核级分析工具，专注单个 kernel 的占用率、效率、带宽等指标     |
| `nvprof`                | 已废弃，建议用 `nsys` 和 `ncu` 替代             |
| `ncu-ui`                | Nsight Compute 的 GUI 前端工具             |

Nsight 分工：

* **Nsight Systems (nsys)**：系统级、高层次分析
* **Nsight Compute (ncu)**：细粒度、内核级性能剖析


---

### 5. **典型 GUI 使用方式**
![image](https://github.com/user-attachments/assets/38b8c65f-bd7f-468a-a4e8-da66aaa7826f)
* **Nsight Systems** 打开 `.nsys-rep`：

  * 菜单：`File → Open → .nsys-rep 文件`
  * 查找 kernel，如 `ampere_sgemm`
  * 使用 “Zoom to Selected” 聚焦分析
  * 右键 → `Profile with Nsight Compute`

* **Nsight Compute** CLI 示例：

```bash
ncu --kernel-name matrixMulKernelOptimized --launch-skip 0 --launch-count 1 --section Occupancy ./nvtx_matmul
```

---

### 6. **权限问题与解决办法**

若运行 `ncu` 遇到权限问题，可修改配置文件：

```bash
sudo nano /etc/modprobe.d/nvidia.conf
# 添加内容：
options nvidia NVreg_RestrictProfilingToAdminUsers=0
sudo reboot
```

---

### 7. **扩展资源与文档推荐**

* [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
* [Stack Overflow](https://stackoverflow.com/)
* [NVIDIA Nsight Documentation](https://docs.nvidia.com/nsight/)
* PyTorch Triton 相关优化文档
* 可搭配 ChatGPT/LLM 搜索概念辅助理解

---

以下是关于 CUDA Kernel Profiling 的介绍内容，**专注于工具用途、命令、原理，不包含任何分析性结论或优化建议**：

---

## 🎯 CUDA Kernel Profiling 工具与方法概述

### 1. **Nsight Compute (ncu) Kernel Profiling**

**命令行示例：**

```bash
ncu --kernel-name matrixMulKernelOptimized --launch-skip 0 --launch-count 1 --section Occupancy ./nvtx_matmul
```

* `--kernel-name`：指定分析的 kernel 名称。
* `--launch-skip`：跳过前面指定数量的 kernel 启动。
* `--launch-count`：分析多少个 kernel 启动。
* `--section Occupancy`：收集 Occupancy 相关指标。
* `./nvtx_matmul`：待执行的可执行程序。

**用途：**
用于详细分析指定 kernel 的性能，包括占用率（occupancy）、内存访问模式、执行效率等低层级细节。

---

### 2. **Vector Addition Kernel Profiling**

![image](https://github.com/user-attachments/assets/6dbd6ff2-2c63-4f17-bd64-df56610137bb)
![image](https://github.com/user-attachments/assets/a7633599-a1cf-4d5f-80db-6b48912ac75d)
![image](https://github.com/user-attachments/assets/375ac29e-e417-4184-9e1f-b1c6d83f360d)

当使用 2^25（即约 3350 万）个元素进行向量加法时，通常会测试多种 kernel 实现方式，包括：

* 基础实现（无 block 或 thread）
* 网格 + 线程版本
* 使用共享内存或 loop unrolling 的优化版本

这种设置常用于对比不同实现间的性能差异。

---

### 3. **NVTX（NVIDIA Tools Extension Library）**

**基本用途：**
用于在代码中手动插入标记，以帮助 profiler 工具（如 `nsys`, `ncu`）精确定位某些代码段。

**编译示例：**

```bash
nvcc -o matmul matmul.cu -lnvToolsExt
```

**运行并生成 profile 报告：**

```bash
nsys profile --stats=true ./matmul
```

**查看生成的统计信息：**

```bash
nsys stats report.qdrep
```

---

### 4. **CUPTI：CUDA Profiling Tools Interface**

**功能说明：**
CUPTI 提供一组 API，用于构建自定义 profiler 工具，支持细粒度的事件收集与跟踪。

**提供的 API 接口包括：**

| 接口类别            | 功能简介                                 |
| --------------- | ------------------------------------ |
| Activity API    | 收集 GPU 活动事件，如 kernel 启动、memcpy 等     |
| Callback API    | 插桩并收集 CUDA Runtime 与 Driver API 调用信息 |
| Event API       | 收集硬件计数器事件（如指令数、缓存命中等）                |
| Metric API      | 提供高级抽象性能指标（如 FLOP/s、带宽利用率等）          |
| Profiling API   | 用于控制 profiling 会话（例如选择要分析的 kernel）   |
| PC Sampling API | 对程序计数器进行采样，用于指令热度图等分析                |
| SASS Metric API | 汇编级指令统计分析                            |
| Checkpoint API  | 用于阶段性保存 profiling 状态                 |

**官方文档：**
[CUPTI Overview - NVIDIA Docs](https://docs.nvidia.com/cupti/overview/overview.html)

**备注：**
由于 CUPTI 使用复杂、学习曲线陡峭，课程中通常以 `nsys` 与 `ncu` 为主工具，CUPTI 适合对 profiler 有更高自定义需求的开发者。

---

### 5. **Profiler 工具定位对比**

| 工具      | 层级       | 适用对象                   | 是否 GUI 支持     |
| ------- | -------- | ---------------------- | ------------- |
| `nsys`  | 系统级      | 多 kernel、整体程序          | ✅（nsight-sys） |
| `ncu`   | Kernel 级 | 单个 kernel 的详细分析        | ✅（ncu-ui）     |
| `CUPTI` | 底层接口     | 自定义 profiler 工具开发      | ❌（开发者实现）      |
| `nvtx`  | 标注辅助     | 精细标记事件范围，配合 `nsys/ncu` | 否             |

---

本节仅介绍相关 profiler 工具与使用方式，供 CUDA 内核分析与性能优化中参考使用。

