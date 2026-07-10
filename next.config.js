/** @type {import('next').NextConfig} */

let withBundleAnalyzer = require("@next/bundle-analyzer")({
    enabled: process.env.ANALYZE === "true",
});

const nextConfig = {
  output: "export",                    // emit a fully static site into ./out (no Node server needed)
  basePath: "/llm-viz",                // GitHub Pages 项目站部署在 /<repo>/ 子路径下
  trailingSlash: true,                 // 页面地址成 /llm-viz/,相对资源路径才解析得对
  reactStrictMode: false, // Recommended for the `pages` directory, default in `app`.
  productionBrowserSourceMaps: false, // Don't ship source maps to the public internet.
  images: { unoptimized: true },       // static export can't use the on-demand image optimizer
  experimental: {
    appDir: true,
  },
  // NOTE: static export supports neither server-side redirects nor route handlers, so the
  // old `/llm-viz -> /llm` redirect and the `/cpu` RISC-V module (which had a file-write API)
  // were removed. The app is now 100% static / client-side.
};

module.exports = withBundleAnalyzer(nextConfig);
