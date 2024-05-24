
Our goal is to provide the code-based example for how to measure the runtime performance in **2102208 Programming for EE**.

#### Recap

    ```
    There are many ways to measure the runtime performance: 

    1. Time difference is the amount of time spent for the operation (depend on the processor).

    2. MACs and flop counts (independent of the processor)
        - Flop counts is the number of operations, which include addition, subtraction, multiplication, and division operations on floating-point numbers.  
        - MACs, on the other hand, is the number of a set of operations, that is, the multiply-accumulate operations that involves one multiplying two numbers and adding the result. Often, used in deep learning apps. 
    
    3. FLOP/s or floating-point operations per second is the number of operations executed per second by a processor.   
    ```

  
## Python libraries 

We provide a python notebook for you to measure the runtime performance of a 2D convolution operation on an image.

```
Profiling_w_libraries.ipynb
``` 

The file contains the examples of the following topics:

1. Time difference
    - Time library [Ref: https://docs.python.org/3/library/time.html]
    - cProfile library [Ref: https://docs.python.org/3/library/profile.html]
2. Macs and flop-counts 
    - Operation counter (thop) [Ref: https://github.com/Lyken17/pytorch-OpCounter]
    - Torch profiler [Ref: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html]
3. FLOP/S 
    - FLOPS profiler [Ref: https://pypi.org/project/flops-profiler]

## Nsys profiling

Nsys profiling is a very useful tool to evaluate the runtime performance. Here, you can look at our example python script `example_torch_profile.py`. 

Then, all you have to do is running the following command in the command prompt:

```
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --cudabacktrace=true -x true -o my_profile python example_torch_profile.py --force-overwrite true --cuda-memory-usage true
```

Ref : https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59 
 