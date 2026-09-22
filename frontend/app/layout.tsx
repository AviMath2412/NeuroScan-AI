import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NeuroScan AI | Brain Tumor Classification & Explainability",
  description: "Deep Learning Diagnostic Platform for Brain MRI Scans with Grad-CAM Visual Explainability",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full bg-slate-50">
      <body className="min-h-full font-sans antialiased text-slate-900 bg-slate-50">
        {children}
      </body>
    </html>
  );
}
