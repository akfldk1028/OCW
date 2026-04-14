import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  async rewrites() {
    return [
      {
        source: "/api/binance/:path*",
        destination: "https://api.binance.com/api/:path*",
      },
    ];
  },
};

export default nextConfig;
