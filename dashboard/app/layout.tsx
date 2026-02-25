import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "OCW Trading Dashboard",
  description: "Agent-vs-Agent crypto trading engine monitor",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
