# NeRF-NCNN

NeRF implemented by [ncnn](https://github.com/Tencent/ncnn) framework based on C++, reference from [taichi-nerfs](https://github.com/taichi-dev/taichi-nerfs)

***Performance***
|                  | taichi & pytorch | ncnn & c++ |
| ---------------- | ---------------- | ---------- |
| i7-12700+RTX3060 | ~15fps           | ~0.6fps    |


## Demo

![image](./resources/Chair.gif)

![image](./resources/Ficus.gif)

![image](./resources/Hotdog.gif)

![image](./resources/Lego.gif)

![image](./resources/Materials.gif)

![image](./resources/Mic.gif)

![image](./resources/Ship.gif)


## References

1. [ncnn](https://github.com/Tencent/ncnn)
2. [opencv-mobile](https://github.com/nihui/opencv-mobile)
3. [taichi-nerfs](https://github.com/taichi-dev/taichi-nerfs)
4. [vkpeak](https://github.com/nihui/vkpeak)
5. [waifu2x-ncnn-vulkan](https://github.com/nihui/waifu2x-ncnn-vulkan)
