import { build } from "esbuild";
import { compile, compileModule } from "svelte/compiler";
import { readFile } from "node:fs/promises";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../", import.meta.url));
await build({
  entryPoints: [root + "src/server/dashboard/ui/index.js"],
  outfile: root + "src/server/dashboard/static/app.js",
  bundle: true, minify: true, format: "esm", conditions: ["browser"],
  nodePaths: [fileURLToPath(new URL("node_modules", import.meta.url))],
  plugins: [{ name: "svelte", setup(build) {
    build.onLoad({ filter: /\.svelte(?:\.js)?$/ }, async ({ path }) => {
      const source = await readFile(path, "utf8"), options = { filename: path, generate: "client", dev: false };
      const { js, warnings } = path.endsWith(".svelte.js") ? compileModule(source, options) : compile(source, options);
      return { contents: js.code, resolveDir: dirname(path), warnings: warnings.map((w) => ({ text: `${w.code}: ${w.message}` })) };
    });
  } }],
});
