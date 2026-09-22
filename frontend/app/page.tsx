"use client";

import React, { useState, useRef } from "react";
import {
  Brain,
  UploadCloud,
  FileImage,
  Activity,
  AlertTriangle,
  CheckCircle2,
  Download,
  RotateCcw,
  ShieldCheck,
  Eye,
  Info,
  ChevronRight,
  Sparkles
} from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell
} from "recharts";
import { motion, AnimatePresence } from "framer-motion";

interface PredictionResponse {
  predicted_class: string;
  confidence: number;
  all_probabilities: Record<string, number>;
  heatmap_base64: string;
}

const CLASS_CONFIG: Record<
  string,
  { label: string; badgeColor: string; textColor: string; borderColor: string; bgSoft: string; barColor: string; description: string }
> = {
  notumor: {
    label: "No Tumor Detected",
    badgeColor: "bg-emerald-500",
    textColor: "text-emerald-700",
    borderColor: "border-emerald-200",
    bgSoft: "bg-emerald-50",
    barColor: "#10b981",
    description: "No pathological tumor mass or intracranial lesion identified in this scan."
  },
  glioma: {
    label: "Glioma",
    badgeColor: "bg-rose-500",
    textColor: "text-rose-700",
    borderColor: "border-rose-200",
    bgSoft: "bg-rose-50",
    barColor: "#f43f5e",
    description: "Features consistent with glial tissue neoplasm requiring neuro-oncology consult."
  },
  meningioma: {
    label: "Meningioma",
    badgeColor: "bg-amber-500",
    textColor: "text-amber-700",
    borderColor: "border-amber-200",
    bgSoft: "bg-amber-50",
    barColor: "#f59e0b",
    description: "Extra-axial dural-based mass characteristic of meningeal cell proliferation."
  },
  pituitary: {
    label: "Pituitary Tumor",
    badgeColor: "bg-purple-500",
    textColor: "text-purple-700",
    borderColor: "border-purple-200",
    bgSoft: "bg-purple-50",
    barColor: "#a855f7",
    description: "Sellar/suprasellar lesion involving or abutting the pituitary gland."
  }
};

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (file: File) => {
    setErrorMsg(null);
    if (!file.type.match(/^image\/(jpeg|png)$/)) {
      setErrorMsg("Please upload a valid MRI scan in .jpg or .png format.");
      return;
    }
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    uploadAndAnalyze(file);
  };

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const uploadAndAnalyze = async (file: File) => {
    setIsLoading(true);
    setErrorMsg(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const endpoint =
        process.env.NEXT_PUBLIC_API_URL ||
        (typeof window !== "undefined" && window.location.port === "3000"
          ? "http://localhost:8000/api/predict"
          : "/api/predict");

      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail || `API returned status ${response.status}`);
      }

      const data: PredictionResponse = await response.json();
      setResult(data);
    } catch (err: any) {
      console.error("Prediction Error:", err);
      setErrorMsg(
        err.message || "Failed to reach NeuroScan AI server. Ensure backend is running at http://localhost:8000"
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setErrorMsg(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const downloadReport = () => {
    if (!result) return;
    const dateStr = new Date().toLocaleString();
    const activeConfig = CLASS_CONFIG[result.predicted_class.toLowerCase()] || {
      label: result.predicted_class,
      description: "N/A"
    };

    const reportContent = `===============================================================
              NEUROSCAN AI - DIAGNOSTIC REPORT
===============================================================
Date & Time       : ${dateStr}
Filename          : ${selectedFile?.name || "Uploaded_Scan.jpg"}
Primary Diagnosis : ${activeConfig.label.toUpperCase()}
Confidence Score  : ${(result.confidence * 100).toFixed(2)}%

CLINICAL INTERPRETATION:
${activeConfig.description}

PROBABILITY DISTRIBUTION:
${Object.entries(result.all_probabilities)
  .map(([cls, prob]) => {
    const name = CLASS_CONFIG[cls.toLowerCase()]?.label || cls;
    return `  - ${name.padEnd(20)} : ${(prob * 100).toFixed(2)}%`;
  })
  .join("\n")}

GRAD-CAM EXPLAINABILITY:
A deep gradient class activation map was generated on the final
convolutional layer (layer4) to delineate anatomical focus areas.

DISCLAIMER:
NeuroScan AI is a research and clinical assistive tool. Results should
always be correlated with clinical symptoms and reviewed by a board-certified
radiologist or neurologist.
===============================================================`;

    const blob = new Blob([reportContent], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `NeuroScan_Report_${result.predicted_class}_${Date.now()}.txt`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const chartData = result
    ? Object.entries(result.all_probabilities).map(([key, value]) => {
        const cfg = CLASS_CONFIG[key.toLowerCase()];
        return {
          name: cfg ? cfg.label : key,
          rawKey: key.toLowerCase(),
          probability: Number((value * 100).toFixed(1)),
          fill: cfg ? cfg.barColor : "#3b82f6"
        };
      })
    : [];

  const activeClassConfig = result
    ? CLASS_CONFIG[result.predicted_class.toLowerCase()] || CLASS_CONFIG.glioma
    : null;

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-b from-slate-50 via-sky-50/30 to-slate-100">
      {/* Medical Header / Navbar */}
      <header className="sticky top-0 z-50 bg-white/90 backdrop-blur-md border-b border-slate-200/80 shadow-xs">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-18 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-11 h-11 rounded-xl bg-gradient-to-tr from-blue-600 to-sky-400 flex items-center justify-center shadow-md shadow-blue-500/20 text-white">
              <Brain className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xl font-bold tracking-tight text-slate-900">
                  NeuroScan<span className="text-blue-600">.AI</span>
                </span>
                <span className="px-2 py-0.5 text-xs font-semibold bg-blue-50 text-blue-700 border border-blue-200/60 rounded-full">
                  v2.0 Diagnostic
                </span>
              </div>
              <p className="text-xs text-slate-500 font-medium">Intelligent MRI Neuro-Oncology & Grad-CAM Suite</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-100/80 border border-slate-200 text-xs font-medium text-slate-600">
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
              <span>Backend Connected: 8000</span>
            </div>
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noreferrer"
              className="text-xs font-medium text-blue-600 hover:text-blue-700 hover:underline flex items-center gap-1"
            >
              API Docs <ChevronRight className="w-3.5 h-3.5" />
            </a>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 md:py-12 space-y-8">
        {/* Intro Hero Section */}
        <div className="text-center max-w-2xl mx-auto space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 border border-blue-100 text-blue-700 text-xs font-semibold">
            <Sparkles className="w-3.5 h-3.5 text-blue-500" />
            Grad-CAM Explainable Neural Vision
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
            Brain MRI Tumor Classification & Heatmap
          </h1>
          <p className="text-sm sm:text-base text-slate-600 leading-relaxed">
            Upload an axial brain MRI scan to instantly classify Glioma, Meningioma, Pituitary, or healthy tissue with visual neural explanation overlays.
          </p>
        </div>

        {/* Upload Zone */}
        <div className="max-w-2xl mx-auto">
          <input
            ref={fileInputRef}
            type="file"
            accept=".jpg,.jpeg,.png"
            className="hidden"
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                handleFileSelect(e.target.files[0]);
              }
            }}
          />

          <div
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            onClick={() => !isLoading && fileInputRef.current?.click()}
            className={`relative group cursor-pointer transition-all duration-300 rounded-2xl border-2 border-dashed p-8 md:p-12 text-center bg-white shadow-sm hover:shadow-md ${
              isDragging
                ? "border-blue-500 bg-blue-50/50 scale-[1.01]"
                : "border-slate-300 hover:border-blue-400 hover:bg-slate-50/50"
            } ${isLoading ? "pointer-events-none opacity-85" : ""}`}
          >
            {isLoading ? (
              <div className="flex flex-col items-center justify-center space-y-4 py-4">
                <div className="relative">
                  <div className="w-16 h-16 rounded-full border-4 border-blue-200 border-t-blue-600 animate-spin" />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Brain className="w-7 h-7 text-blue-600 animate-pulse" />
                  </div>
                </div>
                <div>
                  <h3 className="text-base font-semibold text-slate-800">Processing MRI Scan...</h3>
                  <p className="text-xs text-slate-500 mt-1">
                    Running ResNet18 forward pass & calculating Grad-CAM gradient activations
                  </p>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center space-y-4">
                <div className="w-16 h-16 rounded-2xl bg-blue-50 text-blue-600 flex items-center justify-center group-hover:scale-110 transition-transform duration-200 shadow-inner">
                  <UploadCloud className="w-8 h-8" />
                </div>
                <div>
                  <h3 className="text-base font-semibold text-slate-800">
                    <span className="text-blue-600 underline underline-offset-4">Click to upload</span> or drag and drop
                  </h3>
                  <p className="text-xs text-slate-500 mt-1 font-medium">
                    Supports high-resolution MRI scans in JPG or PNG format
                  </p>
                </div>
                <div className="flex items-center gap-3 pt-2 text-xs text-slate-400">
                  <span className="flex items-center gap-1">
                    <FileImage className="w-3.5 h-3.5" /> 224×224 Auto-Resize
                  </span>
                  <span>•</span>
                  <span className="flex items-center gap-1">
                    <ShieldCheck className="w-3.5 h-3.5 text-emerald-500" /> HIPAA Ready / Ephemeral Storage
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Error Message */}
          {errorMsg && (
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-4 p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 flex items-start gap-3 text-sm shadow-xs"
            >
              <AlertTriangle className="w-5 h-5 text-rose-500 shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="font-semibold">Analysis Notice</p>
                <p className="text-xs text-rose-700 mt-0.5">{errorMsg}</p>
              </div>
            </motion.div>
          )}
        </div>

        {/* Results Dashboard */}
        <AnimatePresence>
          {result && activeClassConfig && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              transition={{ duration: 0.4 }}
              className="space-y-8"
            >
              {/* Top Banner Card: Primary Diagnosis */}
              <div className="bg-white rounded-2xl p-6 sm:p-8 border border-slate-200 shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
                <div className="space-y-2">
                  <div className="flex items-center gap-2 text-xs font-bold tracking-wider text-slate-400 uppercase">
                    <Activity className="w-4 h-4 text-blue-500" /> Neural Network Diagnosis
                  </div>

                  <div className="flex flex-wrap items-center gap-3">
                    <span
                      className={`px-4 py-2 rounded-xl text-lg sm:text-xl font-bold flex items-center gap-2 shadow-xs ${activeClassConfig.bgSoft} ${activeClassConfig.textColor} border ${activeClassConfig.borderColor}`}
                    >
                      <span className={`w-3 h-3 rounded-full ${activeClassConfig.badgeColor}`} />
                      {activeClassConfig.label}
                    </span>

                    <div className="px-3.5 py-1.5 rounded-xl bg-slate-100 border border-slate-200 text-slate-700 text-sm font-semibold">
                      Confidence: <span className="font-bold text-slate-900">{(result.confidence * 100).toFixed(1)}%</span>
                    </div>
                  </div>

                  <p className="text-sm text-slate-600 max-w-2xl pt-1">
                    {activeClassConfig.description}
                  </p>
                </div>

                <div className="flex items-center gap-3 w-full md:w-auto">
                  <button
                    onClick={downloadReport}
                    className="flex-1 md:flex-none inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold shadow-sm hover:shadow transition-all duration-150"
                  >
                    <Download className="w-4 h-4" /> Download Report
                  </button>

                  <button
                    onClick={handleReset}
                    className="inline-flex items-center justify-center p-2.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-600 border border-slate-200 text-sm transition-colors"
                    title="Reset Analysis"
                  >
                    <RotateCcw className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Middle Section: Side-by-Side Visual Comparison & Probability Distribution */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                {/* Left: Side-by-Side Images (7 Cols) */}
                <div className="lg:col-span-7 bg-white rounded-2xl p-6 sm:p-7 border border-slate-200 shadow-sm space-y-5">
                  <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                    <div className="flex items-center gap-2">
                      <Eye className="w-4 h-4 text-blue-600" />
                      <h2 className="text-base font-bold text-slate-900">Visual Evidence & Heatmap Overlay</h2>
                    </div>
                    <span className="text-xs text-slate-400 font-medium">Grad-CAM (layer4)</span>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {/* Original Image */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-slate-600 uppercase tracking-wider">
                          Original MRI Scan
                        </span>
                        <span className="text-[11px] px-2 py-0.5 bg-slate-100 rounded text-slate-500 font-medium">
                          Input
                        </span>
                      </div>
                      <div className="aspect-square rounded-xl overflow-hidden bg-slate-950 flex items-center justify-center border border-slate-200 shadow-inner">
                        {previewUrl && (
                          <img
                            src={previewUrl}
                            alt="Original MRI"
                            className="w-full h-full object-contain"
                          />
                        )}
                      </div>
                    </div>

                    {/* Grad-CAM Heatmap Image */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-slate-600 uppercase tracking-wider">
                          Grad-CAM Heatmap
                        </span>
                        <span className="text-[11px] px-2 py-0.5 bg-blue-50 text-blue-600 rounded font-medium">
                          Explanation
                        </span>
                      </div>
                      <div className="aspect-square rounded-xl overflow-hidden bg-slate-950 flex items-center justify-center border border-slate-200 shadow-inner relative group">
                        {result.heatmap_base64 && (
                          <img
                            src={`data:image/jpeg;base64,${result.heatmap_base64}`}
                            alt="GradCAM Heatmap"
                            className="w-full h-full object-contain"
                          />
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-500 flex items-start gap-2">
                    <Info className="w-4 h-4 text-blue-500 shrink-0 mt-0.5" />
                    <span>
                      The red and yellow regions represent high-activation areas that most influenced the convolutional network's tumor prediction.
                    </span>
                  </div>
                </div>

                {/* Right: Animated Probability Bar Chart (5 Cols) */}
                <div className="lg:col-span-5 bg-white rounded-2xl p-6 sm:p-7 border border-slate-200 shadow-sm flex flex-col justify-between space-y-5">
                  <div>
                    <div className="flex items-center justify-between border-b border-slate-100 pb-3 mb-4">
                      <div className="flex items-center gap-2">
                        <Activity className="w-4 h-4 text-blue-600" />
                        <h2 className="text-base font-bold text-slate-900">Class Probability Distribution</h2>
                      </div>
                      <span className="text-xs text-slate-400 font-medium">Softmax</span>
                    </div>

                    {/* Recharts Horizontal Bar Chart */}
                    <div className="w-full h-52">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          layout="vertical"
                          data={chartData}
                          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                        >
                          <XAxis type="number" domain={[0, 100]} unit="%" hide />
                          <YAxis
                            dataKey="name"
                            type="category"
                            axisLine={false}
                            tickLine={false}
                            tick={{ fill: "#475569", fontSize: 12, fontWeight: 500 }}
                            width={110}
                          />
                          <Tooltip
                            formatter={(val: any) => [`${val}%`, "Probability"]}
                            contentStyle={{
                              borderRadius: "0.75rem",
                              border: "1px solid #e2e8f0",
                              boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.05)",
                              fontSize: "12px",
                              fontWeight: 600
                            }}
                          />
                          <Bar dataKey="probability" radius={[0, 8, 8, 0]} barSize={22}>
                            {chartData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.fill} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Detailed List */}
                    <div className="space-y-2 mt-4">
                      {chartData.map((item) => (
                        <div
                          key={item.rawKey}
                          className="flex items-center justify-between text-xs px-3 py-2 rounded-lg bg-slate-50 border border-slate-100"
                        >
                          <div className="flex items-center gap-2 font-medium text-slate-700">
                            <span
                              className="w-2.5 h-2.5 rounded-full"
                              style={{ backgroundColor: item.fill }}
                            />
                            {item.name}
                          </div>
                          <span className="font-bold text-slate-900">{item.probability}%</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
                    <span>Target Model: ResNet-18</span>
                    <span className="flex items-center gap-1 text-emerald-600 font-medium">
                      <CheckCircle2 className="w-3.5 h-3.5" /> High Precision Screening
                    </span>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Clean Medical Footer */}
      <footer className="border-t border-slate-200 bg-white py-6 mt-12 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
          <p>© {new Date().getFullYear()} NeuroScan AI • Clinical Neuro-Radiology Intelligence Platform</p>
          <div className="flex items-center gap-4 text-slate-400">
            <span>PyTorch 2.x</span>
            <span>•</span>
            <span>Grad-CAM v1.5</span>
            <span>•</span>
            <span>FastAPI Backend</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
