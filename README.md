# rbf-rs

Recursive (joint) bilateral filtering in rust: a fast approximation of the bilateral filter with performance independent of kernel size.

Rayon-parallelized horizontal filtering + transpose + horizontal filtering + transpose.

Timings:
| Method        | Time (ms) |
| ------------- | --------- |
| Rust RBF (ours)          | 3.14      |
| C++ RBF          | 20.0     |
| libblur BF          | 92.1      |
| OpenCV          | 171      |

NB: (518, 518, 3) image, i7 4.7Ghz, 16GB; k=21, σ_space=75, σ_range=75 for reference BFs


Based on:
- ['Recursive bilateral filtering'](https://doi.org/10.1007/978-3-642-33718-5_29), ECCV12, Qingxiong Yang
- ['recursive-bf'](https://github.com/ufoym/recursive-bf), github, ufoym

## Examples:

```bash
cargo run --release --example filtering tests/data/blobs.jpg tests/out/img_filtered.png 0.1 0.03
```

```bash
cargo run --release --example compare_methods
```

```bash
cargo run --release --example bench_methods
```

## To compile references:

```bash
git clone https://github.com/ufoym/recursive-bf/
mkdir recursive-bf/example/stb
curl -o recursive-bf/example/stb/stb_image.h https://raw.githubusercontent.com/nothings/stb/refs/heads/master/stb_image.h
curl -o recursive-bf/example/stb/stb_image_write.h https://raw.githubusercontent.com/nothings/stb/refs/heads/master/stb_image_write.h
```

```bash
cp benches/bench.cpp recursive-bf/example/bench.cpp
cd recursive-bf/
g++ -std=c++20 example/bench.cpp -o ../benches/rbf_cpp_bench
```

## To build website:

```bash
wasm-pack build --target web
cp -r pkg/ website/
cd website
yarn build
yarn serve
#python3 -m http.server -d website
```


```bash
./build.sh
npx serve dist --config ../serve.json
```
