# Neural-Net-RS
Rust implementation of a toy neural net library
## 🚴 Usage

### 🐑 Use `cargo generate` to Clone this Template

[Learn more about `cargo generate` here.](https://github.com/ashleygwilliams/cargo-generate)

```
cargo generate --git https://github.com/rustwasm/wasm-pack-template.git --name my-project
cd my-project
```

### 🛠️ Build with `wasm-pack build`

```
wasm-pack build --target web
```

### Include in JS
```js
// <script type="module">
import init, { NeuralNet } from 'path/to/this/pkg/neural_net_rs.js';

init().then(() => {
	const nn = new NeuralNet(/* ... */);
	// ...
});

```

## 🔋 Batteries Included

* [`wasm-bindgen`](https://github.com/rustwasm/wasm-bindgen) for communicating
  between WebAssembly and JavaScript.
* [`console_error_panic_hook`](https://github.com/rustwasm/console_error_panic_hook)
  for logging panic messages to the developer console.
