#  RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals -C link-arg=--shared-memory -C link-arg=--import-memory" \
rm -rf pkg target
rm -rf website/pkg
RUSTUP_TOOLCHAIN=nightly RUSTFLAGS="-C target-feature=+atomics,+bulk-memory -Clink-arg=--shared-memory -Clink-arg=--max-memory=1073741824 -Clink-arg=--import-memory -Clink-arg=--export=__wasm_init_tls -Clink-arg=--export=__tls_size -Clink-arg=--export=__tls_align -Clink-arg=--export=__tls_base" \
wasm-pack build --target web
cp -r pkg/ website/
cd website
npm run build