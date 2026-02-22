# rbf-rs

Recursive (joint) bilateral filtering in rust

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

## TODO:

- tests: unit & integration (similarity to rbf.cpp)
- bench:
- bench individual compoentns (horizontal filter, transpose)
- gnuplot results (speed vs filter size)
- performance:
  - allocate once outside rbf (buf_a, buf_b, buf_img)
  - do fused transpose + norm at end
  - maybe give up on transpose and write explict vertical pass ()
  - maybe try and avoid realloacting buffer at end during normalize and use v.retain instead

## To compile references:

```bash
git clone https://github.com/ufoym/recursive-bf/
mkdir recursive-bf/example/stb
curl -o recursive-bf/example/stb/stb_image.h https://raw.githubusercontent.com/nothings/stb/refs/heads/master/stb_image.h
curl -o recursive-bf/example/stb/stb_image_write.h https://raw.githubusercontent.com/nothings/stb/refs/heads/master/stb_image_write.h
cp benches/bench.cpp recursive-bf/example/bench.cpp
cd recursive-bf/
g++ -std=c++20 example/bench.cpp -o ../benches/rbf_cpp_bench
```
