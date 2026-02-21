# rbf-rs

Recursive (joint) bilateral filtering in rust

## Examples:

```bash
cargo run --release --example filtering
```

```bash
cargo run --release --example compare_methods
```

```bash
cargo run --release --example bench_methods
```

## Desiderata

- parrallelized
- allow forwards - backwards averaging
- allow greyscale
- allow joint filtering
- allow generalizing to N axes i.e, x,y,z
- SIMD?

## TODO:

- compile rbf cpp example & add to git (small binary). Make it CLI program that can either: a) filter image and save to file or b) accept bench flag and return time take
- tests: unit & integration (similarity to rbf.cpp)
- bench:
- bench individual compoentns (horizontal filter, transpose)
- compare against a) the cpp implementation b) opencv (write a python program with deps in module comment & run with uv)
- gnuplot results (speed vs filter size)
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
