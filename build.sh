wasm-pack build --target web
cp -r pkg/ website/
python3 -m http.server -d website